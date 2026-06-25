import glob
import os
import os.path as osp
import platform
import shutil
import sys

from setuptools import find_packages, setup

__version__ = None
exec(open("gsplat/version.py", "r").read())

URL = "https://github.com/nerfstudio-project/gsplat"

BUILD_CUDA = os.getenv("BUILD_CUDA", "0") == "1"
WITH_SYMBOLS = os.getenv("WITH_SYMBOLS", "0") == "1"
LINE_INFO = os.getenv("LINE_INFO", "0") == "1"


def _patch_hipify_ignore_glm():
    """Keep the bundled third_party/glm out of torch's hipify on ROCm.

    torch's hipify (via CUDAExtension) walks every .hpp under the build dir and
    the extension include dirs into its file set, then content-rewrites any GLM
    header a source pulls in -- which drops GLM's .inl files (hipify only copies
    .hpp/.h) and mangles GLM's __CUDACC__/__HIP__ compiler detection, breaking
    the build. GLM 1.0.x already detects __HIP__ and compiles verbatim under the
    -x hip pass, so the fix is simply to leave it untouched: add the glm dir to
    hipify's ``ignores`` and drop it from ``header_include_dirs``. The source
    keeps including <glm/...> via -I, resolved against the pristine bundled tree.
    """
    import torch

    if not torch.version.hip:
        return
    from torch.utils.hipify import hipify_python

    glm_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "gsplat", "cuda", "csrc", "third_party", "glm",
    )
    glm_patterns = [os.path.join(glm_dir, "*"), glm_dir + "*"]
    orig_hipify = hipify_python.hipify

    def hipify_no_glm(*args, **kwargs):
        kwargs["ignores"] = list(kwargs.get("ignores", ())) + glm_patterns
        kwargs["header_include_dirs"] = [
            d for d in kwargs.get("header_include_dirs", [])
            if os.path.abspath(d) != os.path.abspath(glm_dir)
        ]
        return orig_hipify(*args, **kwargs)

    hipify_python.hipify = hipify_no_glm


if BUILD_CUDA:
    _patch_hipify_ignore_glm()


def get_ext():
    import torch
    from torch.utils.cpp_extension import BuildExtension

    # On Windows+ROCm, the non-ninja distutils builder cannot handle .hip files
    # (torch hipify renames .cu -> .hip) and does not forward __HIP_PLATFORM_AMD__
    # to MSVC for .cpp files; ninja handles both. Force ninja for that case only,
    # leaving the CUDA build's default (use_ninja=False) untouched.
    use_ninja = sys.platform == "win32" and bool(torch.version.hip)
    return BuildExtension.with_options(no_python_abi_suffix=True, use_ninja=use_ninja)


def get_extensions():
    import torch
    from torch.__config__ import parallel_info
    from torch.utils.cpp_extension import CUDAExtension

    extensions_dir = osp.join("gsplat", "cuda", "csrc")
    # On Windows with ninja, the build runs from a temp directory so relative
    # include paths don't resolve. Use absolute path for the extension dir.
    abs_extensions_dir = osp.abspath(extensions_dir)
    sources = glob.glob(osp.join(extensions_dir, "*.cu")) + glob.glob(
        osp.join(extensions_dir, "*.cpp")
    )
    sources = [path for path in sources if "hip" not in path and "_winhip" not in path]

    if sys.platform == "win32" and torch.version.hip:
        # On Windows+ROCm, compiling .cpp files with MSVC cl.exe (the torch default
        # for .cpp) fails to link: inherited constructors in c10-dllexport classes
        # (e.g. c10::ValueError) are instantiated in the TU but not exported from
        # c10.dll, causing LNK2001. amdclang (hipcc), used for .cu/.hip, handles
        # this correctly. Route each .cpp through hipcc by creating a same-dir
        # _winhip.cu shim (a copy) so relative #includes resolve.
        shim_sources = []
        for s in sources:
            if s.endswith(".cpp"):
                shim = s[:-4] + "_winhip.cu"
                shutil.copyfile(s, shim)
                shim_sources.append(shim)
            else:
                shim_sources.append(s)
        sources = shim_sources

    undef_macros = []
    define_macros = []

    if sys.platform == "win32":
        define_macros += [("gsplat_EXPORTS", None)]

    extra_compile_args = {"cxx": ["-O3"]}
    if not os.name == "nt":  # Not on Windows:
        extra_compile_args["cxx"] += ["-Wno-sign-compare"]
    extra_link_args = [] if (WITH_SYMBOLS or sys.platform == "win32") else ["-s"]

    info = parallel_info()
    if (
        "backend: OpenMP" in info
        and "OpenMP not found" not in info
        and sys.platform != "darwin"
    ):
        extra_compile_args["cxx"] += ["-DAT_PARALLEL_OPENMP"]
        if sys.platform == "win32":
            extra_compile_args["cxx"] += ["/openmp"]
        else:
            extra_compile_args["cxx"] += ["-fopenmp"]
    else:
        print("Compiling without OpenMP...")

    # Compile for mac arm64
    if sys.platform == "darwin" and platform.machine() == "arm64":
        extra_compile_args["cxx"] += ["-arch", "arm64"]
        extra_link_args += ["-arch", "arm64"]

    nvcc_flags = os.getenv("NVCC_FLAGS", "")
    nvcc_flags = [] if nvcc_flags == "" else nvcc_flags.split(" ")
    nvcc_flags += ["-O3"]
    if LINE_INFO:
        nvcc_flags += ["-lineinfo"]
    if torch.version.hip:
        # USE_ROCM was added to later versions of PyTorch.
        # Define here to support older PyTorch versions as well:
        define_macros += [("USE_ROCM", None)]
        undef_macros += ["__HIP_NO_HALF_CONVERSIONS__"]
        # GLM's operator[] bounds checks expand to assert() -> __assert_fail, a
        # __host__ function referenced from GLM's __host__ __device__ accessors.
        # nvcc tolerates this (it supplies a device assert); ROCm clang rejects it.
        # NDEBUG makes the debug-only bounds asserts no-ops (the indices here are
        # compile-time constants, always in range) -- the standard release define.
        nvcc_flags += ["-DNDEBUG"]
        # ROCm clang's fast-math is more aggressive than CUDA's --use_fast_math
        # and perturbs ill-conditioned projection/covariance gradients past the
        # upstream test tolerances, so keep it OFF by default (opt-in FAST_MATH=1).
        if os.getenv("FAST_MATH", "0") == "1":
            nvcc_flags += ["-ffast-math"]
    else:
        nvcc_flags += ["--use_fast_math", "--expt-relaxed-constexpr"]
    extra_compile_args["nvcc"] = nvcc_flags
    if sys.platform == "win32":
        extra_compile_args["nvcc"] += ["-DWIN32_LEAN_AND_MEAN"]

    extension = CUDAExtension(
        f"gsplat.csrc",
        sources,
        include_dirs=[abs_extensions_dir],
        define_macros=define_macros,
        undef_macros=undef_macros,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )

    return [extension]


setup(
    name="gsplat",
    version=__version__,
    description=" Python package for differentiable rasterization of gaussians",
    keywords="gaussian, splatting, cuda",
    url=URL,
    download_url=f"{URL}/archive/gsplat-{__version__}.tar.gz",
    python_requires=">=3.7",
    install_requires=[
        "ninja",
        "numpy",
        "jaxtyping",
        "rich>=12",
        "torch",
        "typing_extensions; python_version<'3.8'",
    ],
    extras_require={
        # dev dependencies. Install them by `pip install gsplat[dev]`
        "dev": [
            "black[jupyter]==22.3.0",
            "isort==5.10.1",
            "pylint==2.13.4",
            "pytest==7.1.2",
            "pytest-xdist==2.5.0",
            "typeguard>=2.13.3",
            "pyyaml==6.0",
            "build",
            "twine",
        ],
    },
    ext_modules=get_extensions() if BUILD_CUDA else [],
    cmdclass={"build_ext": get_ext()} if BUILD_CUDA else {},
    packages=find_packages(),
    # https://github.com/pypa/setuptools/issues/1461#issuecomment-954725244
    include_package_data=True,
)
