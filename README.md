![Glow Logo](./docs/logo.svg)

[![pytorch](https://circleci.com/gh/pytorch/glow.svg?style=shield)](https://circleci.com/gh/pytorch/glow)


Glow is a machine learning compiler and execution engine for hardware
accelerators.  It is designed to be used as a backend for high-level machine
learning frameworks.  The compiler is designed to allow state of the art
compiler optimizations and code generation of neural network graphs. This
library is in active development. The project plan is described in the Github
issues section and in the
[Roadmap](https://github.com/pytorch/glow/wiki/Glow-Roadmap) wiki page.


Glow是一个深度学习编译器，它主要的工作就是为了提升深度学习框架训练出的model在不同的硬件平台上的表现，可以将我们通过pytorch，caffe2训练好的model导入到glow里，然后通过编译优化技术，产生部署到不同的hardware accelerator上。Glow实际上是存在两个IR,一个IR命名为高阶IR，一个则为低阶IR。高阶IR实际上是整个stack共享的部分，因此对该部分的优化则表现为一些通用性的优化。相比之下，低阶IR则更加specific，对其的优化更加关注于hardware platforms本身的架构设计。那么为什么能叫glow，glow其实是graph+lowering的简写，意思就是说，通过这个低阶的IR，从而在针对大量不同的上层model中的op到下层不同hardware accelerator的实现都尽可能通过一些比较简单的线性代数源语来实现，有点类似精简指令集的感觉。

为什么要做deep learning compiler，其实motivation很简单。我们现在经常用的深度学习framework，像TensorFlow，pytorch，mxnet都是将我们用内部api搭好的nn建模成了一个computational graph，然后对图中的每个node进行一次执行，得到最终的结果。然而，这样遍历图的方式是非常低效的，那么，我们就可以通过将生成的计算图扔进编译器，然后通过某些编译优化的手段，对整个计算图进行optimization，然后再执行这张经过optimization过后的graph，此时的效率就可以得到很大的提高。那么glow存在的意义其实就是为了弥补hardware到framework之间的gap。这样的好处显而易见，hardware的engineer可以更加关注硬件的design，而上层模型的engineer可以尽可能复杂的实现model的feature，glow则可以将software和hardware有机的结合起来，这其实也就是经常大家所说的software-hardware co-design，glow能够根据硬件架构的不同设计，来做一些任务，比如instruction的selection。memory的allocation和相关computational graph的scheduling。

XLA: XLA的做法实际上是将从deep learning framework训练得到的computational graph中的每个node都抽象成为最基本的线性代数操作，然后调用目前经过专家手工优化的库比如说，cpu上的eign，gpu上的cudnn来提交表现性能。比如，对于dot product这个op，xla则采用了向量化的LLVM IR并且上层仅仅针对TensorFlow下层针对TPU，开源的代码不多，仅仅知道个大概即可。

TVM: TVM则是通过将computational graph中的node结合loopy进行优化（因为在deep learning中，大部分需要我们优化的工作都是多重循环）lower到halide IR，然后通过Halide IR来做cuda，opencl，metal等不同backend的支持。

Tensor Comprehensions: TC其实是为神经网络提供了一个新的abstraction，以至于让JIT这样的compiler可以通过一定的算法来找到最优的执行plan，然后这个plan又被根据你指定的不同backend来generate成不同的code，其实TC的好处很明显，就是能够帮我们找到一些现在不存在的op，并且通过将其高效的实现出来。

Glow:

Glow的思路很简单，和上述这些deep learning compiler一样， 有一个或多个IR，然后在低阶IR中，glow会将复杂的op通过一系列简单的线性代数源语来实现。

关于Glow的motivation其实也是很简单的，也就是说，在得到一张computational graph后，我们仅仅通过一层编译手段，将graph中的每个op都变成由一系列loop和其他低阶IR这样的优化显然是不够的。我们还必须有考虑到高阶的IR。比如对于一个多重for-loop语句来看，我们不能通过一High-Level IR: 实际很简单，就是我们通过framework得到的computational graph，针对输入的不同shape和不同data type的data，我们有专门的node来处理他们，针对不同batch size的data，我们可以构建多个glow graph来通过jit对其进行re-compute。其中包括一些storage node，constant node， placeholder node个传统的编译器来帮我们解决这个问题，多层for-loop的优化，他们是做不到的。此时，针对这个多重for-loop（卷积）我们就可以定义一种高阶的IR，例如将data的format定义为tensor（N, C, H, W）的格式，从而帮我们完成相应的optimization。有了这个motivation，glow就被设计出来了，只要让compiler的前几个stage是target-independent的，让他更加倾向于我们所需要解决的任务的data type就行。但是当compiler越接近底层的不同hardware platforms的时候，我们的低阶IR就要更加specific到硬件架构的设计了。

## High-Level IR: 

实际很简单，就是我们通过framework得到的computational graph，针对输入的不同shape和不同data type的data，我们有专门的node来处理他们，针对不同batch size的data，我们可以构建多个glow graph来通过jit对其进行re-compute。其中包括一些storage node，constant node， placeholder node



## Node Lowering: 

经过前面的讲解，我们知道，当出现一个新的hardware的时候，如果我们将现有的model部署到这个新的hardware上，我们就要去重新在这个hardware上写一套指令集(ISA)来支持该model中是所有op。如果此时出现了一个新的op，那么如果要将该op部署到不同的hardware platforms上，我们又得去在每个hardware上都去实现这个新op，随着op和hardware数量的增长，完成该任务的工作量是巨大的。那么，glow是怎么做的呢？glow的做法并不是直接去compile这个high-level的op，而是将这个high-level的op分解成为多个不同的low-level的线性代数源语。比如就拿nn中最常见的fully connected layer来说，glow并不会直接去compile一个fully connected layer的op，而是将该op分解成为matrix multiplication和broadcast add的两个简单的线性代数源语。按照这个思路类推，不管有多少high-level复杂的op，都可以被node lowering分解成多个简单的这样的源语。添加node lowering的好处其实有很多：


1. 可以为该图添加额外的图级别的优化。

2. 也许会影响相关指令的调度。

3. 对于具体的backend表现出特定的优化。


## Lower-Level IR: 

在一张完整的computational graph在经过high-level的优化，然后再通过node lowering变成一系列简单的线性代数源语后，就得通过glow中的IRGen( IR Generation)来做CodeGen了。因为在一个编译器中，IRGen起的就是一种one-to-many的作用，将一种语言翻译成为多种machine code的过程。并且low-level的IR能够使得一些在high-level的IR不能做的optimization，在这个阶段都可以实现。就比如说，当我们有一条指令需要通过address来操作某个data到时候，在high-level是根本没办法直接访问到memory的，那么，这时候只能在low-level来做这种关于memory的optimization了。low-level的IR被设计成为了in-memory的格式，在IR中的function都会有两个特别重要的部分，一个部分叫做declare，相当于声明所有变量，有点类似C语言中的全局变量。另外一个部分叫做program，由一系列的instruction构成，往往用来初始化变量和告诉程序要做些什么事情。


## 整个compilation flow总结一下：

1. 我们通过glow自带的graph loader将在deep learning framework上train出来的model载入到glow。

2. 如果你想对整张图进行training，那么你可以设置每个node为differentiated的。

3. 通过high-level IR的optimization。

4. 通过node lowering，将high-level层面的node进行break down，成为简单的线性代数基本源语。

5. IRGen将low-level IR 转换成一系列的instruction。

6. 在进行low-level IR的optimization。

7. 最终得到backend上相应的代码和backend具体的优化。



## Partners

Contributions to Glow are welcomed and encouraged! Glow is developed in
collaboration with the following partners:


<!---
Note:
List of partner logos sorted alphabetically column order.
-->

| ![Bitmain Logo](./docs/partners/bitmain.png) | ![Habana Logo](./docs/partners/habana.png) | ![ST Logo](./docs/partners/st.png)  |
:-------------------------:|:-------------------------:|:-------------------------:
| ![Cadence Logo](./docs/partners/cadence.png) | ![Intel Logo](./docs/partners/intel.png) | ![Synopsys Logo](./docs/partners/synopsys.png) |
| ![CEVA Logo](./docs/partners/ceva.png)   |  ![Marvell Logo](./docs/partners/marvell.png) |  |
| ![Esperanto Logo](./docs/partners/esperanto.png)  | ![NXP Logo](./docs/partners/nxp.png) |  |


## How does it work?

Glow lowers a traditional neural network dataflow graph into a two-phase
strongly-typed [intermediate representation (IR)](./docs/IR.md). The high-level
IR allows the optimizer to perform domain-specific optimizations. The
lower-level instruction-based address-only IR allows the compiler to perform
memory-related optimizations, such as instruction scheduling, static memory
allocation and copy elimination. At the lowest level, the optimizer performs
machine-specific code generation to take advantage of specialized hardware
features. Glow features a lowering phase which enables the compiler to support a
high number of input operators as well as a large number of hardware targets by
eliminating the need to implement all operators on all targets. The lowering
phase is designed to reduce the input space and allow new hardware backends to
focus on a small number of linear algebra primitives.
The design philosophy is described in an [arXiv paper](https://arxiv.org/abs/1805.00907).

![](./docs/3LevelIR.png)

## Getting Started

### System Requirements

Glow builds and runs on macOS and Linux. The software depends on a modern C++
compiler that supports C++11, on CMake, LLVM (>=7.0), glog, protocol buffers, and
libpng.

#### Get Glow!

  ```bash
  git clone git@github.com:pytorch/glow.git  # or: git clone https://github.com/pytorch/glow.git
  cd glow
  ```

#### Submodules

Glow depends on a few submodules: googletest, onnx, and a library
for FP16 conversions.

To get them, from the glow directory, run:

  ```bash
  git submodule update --init --recursive
  ```

#### Source dependencies

Glow depends on `fmt`, which must be built from source:
```bash
git clone https://github.com/fmtlib/fmt
mkdir fmt/build
cd fmt/build
cmake ..
make
sudo make install
```

#### macOS

Install the required dependencies using either [Homebrew](https://brew.sh/) or
[MacPorts](https://www.macports.org/). If using Homebrew, run:

  ```bash
  brew install cmake graphviz libpng ninja protobuf wget glog autopep8 llvm   \
      boost double-conversion gflags jemalloc libevent lz4 openssl pkg-config \
      snappy xz
  ```

If using MacPorts, run:

  ```bash
  port install cmake graphviz libpng ninja protobuf-cpp wget google-glog \
      boost double-conversion gflags jemalloc libevent lz4 openssl snappy xz
  # Choose version >= 7
  export LLVM_VERSION=7
  port install llvm-$LLVM_VERSION.0 
  ```


Note that LLVM is installed in a non-default location to avoid conflicts with
the system's LLVM --Homebrew usually installs LLVM in `/usr/local/opt/llvm/`,
whereas MacPorts installs it in `/opt/local/libexec/llvm-$LLVM_VERSION.0/`. This means that
CMake will need to be told where to find LLVM when building; instructions on
that can be found [here](#building-with-dependencies-llvm).

Finally, create a symbolic link to the Homebrew- or MacPorts-installed
`clang-*` tools so that the `utils/format.sh` script is able to find them later
on. For a Homebrew-managed installation, run:
  ```
  ln -s "/usr/local/opt/llvm/bin/clang-format" "/usr/local/bin/clang-format"
  ln -s "/usr/local/opt/llvm/bin/clang-tidy" "/usr/local/bin/clang-tidy"
  ```
For MacPorts, run:
  ```
  ln -s "/opt/local/libexec/llvm-$LLVM_VERSION.0/bin/clang-format" "/usr/local/bin/clang-format"
  ln -s "/opt/local/libexec/llvm-$LLVM_VERSION.0/bin/clang-tidy" "/usr/local/bin/clang-tidy"
```

> **Note:** Starting with macOS Mojave, Xcode's command line tools changed header layout. 
> In order for Glow to build on Mojave, you might need to install
> `macOS_SDK_headers_for_macOS_10.14.pkg`, located in 
> `/Library/Developer/CommandLineTools/Packages/`.
> For macOS Catalina you might need to explicitly specify SDKROOT: 
> `export SDKROOT="/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk"`


#### Ubuntu

[The following instructions have been tested on Ubuntu 16.04 and 18.04]

In order to build Glow on Ubuntu it is necessary to install a few packages. The
following command should install the required dependencies:

  ```bash
  sudo apt-get install clang clang-8 cmake graphviz libpng-dev \
      libprotobuf-dev llvm-8 llvm-8-dev ninja-build protobuf-compiler wget \
      opencl-headers libgoogle-glog-dev libboost-all-dev \
      libdouble-conversion-dev libevent-dev libssl-dev libgflags-dev \
      libjemalloc-dev libpthread-stubs0-dev
  ```

[Note: Ubuntu 16.04 and 18.04 ship with llvm-6 and need to be upgraded before building Glow. Building Glow on Ubuntu 16.04 with llvm-7 fails because llvm-7 xenial distribution uses an older c++ ABI, however building Glow on Ubuntu 18.04 with llvm-7 has been tested and is successful]

It may be desirable to use `update-alternatives` to manage the version of
clang/clang++:

  ```bash
  sudo update-alternatives --install /usr/bin/clang clang \
      /usr/lib/llvm-8/bin/clang 50
  sudo update-alternatives --install /usr/bin/clang++ clang++ \
      /usr/lib/llvm-8/bin/clang++ 50
  ```

Glow uses the system default C/C++ compiler (/usr/bin/c++), and so you may also
want to switch your default C/C++ compiler to clang:

  ```bash
  sudo update-alternatives --config cc
      # Select the option corresponding to /usr/bin/clang ...
  sudo update-alternatives --config c++
      # Select the option corresponding to /usr/bin/clang++ ...
  ```

Glow *should* build just fine with gcc (e.g. gcc 5.4), but we mostly use clang
and are more attentive to compatibility with clang.

Finally, in order to support the ONNX net serialization format, Glow requires
`protobuf >= 2.6.1`, but the above command may install older
version on older Ubuntu (e.g. 14.04). If this is the case, we suggest to look
at `utils/install_protobuf.sh` to install a newer version from source.

For details on installing OpenCL on Ubuntu please see
[these instructions](docs/Building.md#opencl-on-ubuntu).

### Configure and Build

To build the compiler, create a build directory and run cmake on the source
directory. It's a good idea to build two configurations (Release and Debug)
because some programs take a really long time to run in Debug mode. It's also a
good idea to build the project outside of the source directory.

  ```bash
  mkdir build_Debug
  cd build_Debug
  cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug ../glow
  ninja all
  ```

It's possible to configure and build the compiler with any CMake generator,
like GNU Makefiles, Ninja and Xcode build.

For platform-specific build instructions and advanced options, such as
building with Address-Sanitizers refer to this guide:
[Building the Compiler](docs/Building.md).

If you're running macOS v10.14 (Mojave) and `ninja all` fails because it can't
find headers (e.g. `string.h`), run this command to fix it, and try again.
More information is available [here](https://developer.apple.com/documentation/xcode_release_notes/xcode_10_release_notes)
under "Command Line Tools".

  ```bash
  open /Library/Developer/CommandLineTools/Packages/macOS_SDK_headers_for_macOS_10.14.pkg
  ```

For macOS v10.15 (Catalina) you might need to explicitly specify SDKROOT:

   ```bash
   export SDKROOT="/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk"
   ```


#### Building with dependencies (LLVM)

By default, Glow will use a system provided LLVM.  Note that Glow requires LLVM
7.0 or later. If you have LLVM installed in a non-default location (for
example, if you installed it using Homebrew on macOS), you need to tell CMake
where to find llvm using `-DLLVM_DIR`. For example, if LLVM were
installed in `/usr/local/opt`:

  ```bash
  cmake -G Ninja ../glow \
      -DCMAKE_BUILD_TYPE=Debug \
      -DLLVM_DIR=/usr/local/opt/llvm/lib/cmake/llvm
  ```

If LLVM is not available on your system you'll need to build it manually.  Run
the script '`/utils/build_llvm.sh` to clone, build and install LLVM in a local
directory. You will need to configure Glow with the flag `-DLLVM_DIR` to tell
the build system where to find LLVM given the local directory you installed it
in (e.g. `-DLLVM_DIR=/path/to/llvm_install/lib/cmake/llvm` if using
`build_llvm.sh`).

## Testing and Running

### Unit tests

The project has a few unit tests in the tests/unittests subdirectory. To run all
of them, simply run `ninja test`.

### C++ API examples

A few test programs that use Glow's C++ API are found under the `examples/`
subdirectory. The `mnist`, `cifar10`, `fr2en` and `ptb` programs train and run digit
recognition, image classification and language modeling benchmarks,
respectively.

To run these programs, build Glow in Release mode, then run the following commands
to download the cifar10, mnist and ptb databases.

  ```bash
  python ../glow/utils/download_datasets_and_models.py --all-datasets
  ```

Now run the examples. Note that the databases should be in the current working
directory.

  ```bash
  ./bin/mnist
  ./bin/cifar10
  ./bin/fr2en
  ./bin/ptb
  ./bin/char-rnn
  ```

If everything goes well you should see:
  * `mnist`: pictures from the mnist digits database
  * `cifar10`: image classifications that steadily improve
  * `fr2en`: an interactive French-to-English translator
  * `ptb`: decreasing perplexity on the dataset as the network trains
  * `char-rnn`: generates random text based on some document

Note that the default build mode is `Debug`, which means that the compiler
itself is easy to debug because the binary contains debug info, lots of
assertions, and the optimizations are disabled. It also means that the compiler
and runtime are very slow, and the execution time can be hundreds of times
slower than that of release builds. If you wish to benchmark the compiler, run
long benchmarks, or release the product then you should compile the compiler in
Release mode. Check the main CMake file for more details.

More details on testing and running Glow can be found in:
[Testing the Glow Compiler](docs/Testing.md).

### Ahead-of-time Compilation

Glow can be used to compile neural networks into object files containing native
code.  We provide resnet50 (both quantized and non-quantized versions) as an
example of this capability in `examples/bundles/resnet50`.  See [Creating
Standalone Executable Bundles](docs/AOT.md) for more detail.

## Contributing

To get started contributing, please refer to the following guides:
* [Contributing](CONTRIBUTING.md)
* [Coding Standards](docs/CodingStandards.md)
* [Code of Conduct](CODE_OF_CONDUCT.md)

### Communication

* Forums: discuss implementations, research, etc: https://discuss.pytorch.org/c/glow.
  Make sure to label topic with the ["glow"](https://discuss.pytorch.org/c/glow) category.
* GitHub issues: bug reports, feature requests, install issues, RFCs, thoughts, etc.

## License

Glow is licensed under the [Apache 2.0 License](LICENSE).
