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

XLA（Accelerated Linear Algebra，加速线性代数）是一种优化TensorFlow计算的编译器。 XLA（加速线性代数）是为优化TensorFlow计算的特定领域编译器。它可以提高服务器和移动平台的速度、改进内存使用、提升便携性（或者说是可移植性）。 XLA框架是目前处于开发中的实验性项目。

XLA有几个与目标硬件无关的优化，例如公共子表达式消除（CSE，common subexpression elimination），过程与硬件后端无关的（target-independent）算子融合，以及用于为计算而在运行时做内存分配的缓冲区分析。硬件目标无关的步骤之后，XLA将HLO计算发送到后端去做进一步的HLO级优化，HLO级优化会针对硬件目标特定信息和需求，如XLA GPU后端的算子融合、确定如何将计算划分为流。此阶段，后端还可以将某些算子或其组合与优化的库调用进行模式匹配。下一步是特定于目标硬件的代码生成。CPU包括XLA使用的GPU后端会使用LLVM生成低级IR、优化和代码生成。硬件后端以有效方式发出了表示XLA HLO计算所必需的LLVM IR，然后调用LLVM从LLVM IR发出原生代码。GPU后端目前通过LLVM NVPTX后端支持NVIDIA GPU，CPU后端支持多个CPU指令级架构（ ISA）


TVM: TVM则是通过将computational graph中的node结合loopy进行优化（因为在deep learning中，大部分需要我们优化的工作都是多重循环）lower到halide IR，然后通过Halide IR来做cuda，opencl，metal等不同backend的支持。

Tensor Comprehensions: TC其实是为神经网络提供了一个新的abstraction，以至于让JIT这样的compiler可以通过一定的算法来找到最优的执行plan，然后这个plan又被根据你指定的不同backend来generate成不同的code，其实TC的好处很明显，就是能够帮我们找到一些现在不存在的op，并且通过将其高效的实现出来。

Glow:

Glow的思路很简单，和上述这些deep learning compiler一样， 有一个或多个IR，然后在低阶IR中，glow会将复杂的op通过一系列简单的线性代数源语来实现。

关于Glow的motivation其实也是很简单的，也就是说，在得到一张computational graph后，我们仅仅通过一层编译手段，将graph中的每个op都变成由一系列loop和其他低阶IR这样的优化显然是不够的。我们还必须有考虑到高阶的IR。比如对于一个多重for-loop语句来看，我们不能通过一High-Level IR: 实际很简单，就是我们通过framework得到的computational graph，针对输入的不同shape和不同data type的data，我们有专门的node来处理他们，针对不同batch size的data，我们可以构建多个glow graph来通过jit对其进行re-compute。其中包括一些storage node，constant node， placeholder node个传统的编译器来帮我们解决这个问题，多层for-loop的优化，他们是做不到的。此时，针对这个多重for-loop（卷积）我们就可以定义一种高阶的IR，例如将data的format定义为tensor（N, C, H, W）的格式，从而帮我们完成相应的optimization。有了这个motivation，glow就被设计出来了，只要让compiler的前几个stage是target-independent的，让他更加倾向于我们所需要解决的任务的data type就行。但是当compiler越接近底层的不同hardware platforms的时候，我们的低阶IR就要更加specific到硬件架构的设计了。

神经网络编译器：

 神经网络编译器大概有TVM/Glow/TensorRT/TensorComprehension/XLA/Tiramisu。这些针对的都是神经网络模型推理阶段的优化，是从神经网络模型到机器代码的编译。
 
 一般过程是 
 
 神经网络模型->图优化(High-Level IR优化)->中间代码生成（例如Halide)->中间代码优化（例如TC/Tiramisu使用多面体模型进行变换，Lower-Level IR优化 ）->机器代码。
 
 它编译的是神经网络的模型，优化的是网络模型本身，各层数据数据存储的方式（如分块存储，nchw，nhcw），以及各个算子（如mlp，conv）的计算方式（如向量化，分块）等等传统编译器（GCC，Clang这些）的编译范围更广，是从源代码到机器代码的编译，

首先是神经网络编译器丛中间代码到机器代码的过程可能就对应了传统编译器的整个编译过程，比如Halide->机器代码然后他们的目标都是都要针对目标处理器进行的优化。无论是什么代码/模型，最后的优化无非就是如何最大化利用硬件，比如cache的命中率，计算速度啥的，最终目标都是生成好的机器代码。






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

## 深度学习编译器存在的价值是什么？

如果自动生成的代码效果特别好，那直接用就是，不用再讨论价值。现状是，还不够好，我们要对它的价值讨论清楚才能决定是否去投入，当然如果是纯兴趣驱动，我们就不需要这么功利。（我认为，做任何事情之前都要思考且理性下决定，人生苦短，要把时间精力花的更有意义，不废话了）。我思考的结果是：深度学习编译器的价值取决于AI芯片的前途。AI芯片上开发编译器的难度不高，基本上和在GPU上调用cublas, cudnn写程序差不多，因为基本的张量运算都用专用电路固化了，没啥可优化的（当然访存和计算做流水还是要做的），为某款AI芯片研发深度学习编译器，可能只需要关注第一阶段的问题(HLO)，不需要解决第二阶段的问题(codegen)。如果对专用芯片上代码怎么写感兴趣，可参照Glow, 它提供了一个为**Habana 后端，这可能是唯一一个开源的AI芯片代码示例。** 讨论这些是想说明什么问题呢？那就是，假如未来AI芯片一统天下，深度学习编译器不难，也不是那么必要（当然像XLA那样做HLO层面的优化还是有好处的）。只有当通用处理器比较流行且多种处理器并存时，譬如CPU和GPU，也就是当下的状态，深度学习编译器如果能实现其目的就有价值，当然难度也很大。当然，编译器做的好，AI芯片可能就更好做，这就是软硬协同设计的意义。好吧，深度学习编译器的命运与AI芯片竞争格局息息相关，但我们并没有讨论AI芯片未来如何，这是另一个问题了，真正投入搞AI芯片的玩家对这个问题想的更清楚。



## 前端

1，前端用DSL还是限定一个算子集合。XLA没有DSL,而是限定了一些基本算子，element-wise, map, reduce, broadcast, matmul 等等，蛮像函数式编程里面那些基本运算，用这些基本算子可以搭建起来tensorflow 里绝大多数更上层的算子，譬如batchnorm, softmax等。这么做当然限制了表示能力，但却给第二阶段codegen 带来极大便利，因为它只需要为这些限定的算子emit LLVM IR, 用固定的套路即可，相当于逃避了第二阶段优化那个难题。Glow实际上采用了和XLA相似的策略，对于第二步取了个巧，对于常见的矩阵运算算子提供了代码生成的模板，用模板并不意味着对不同的参数（譬如矩阵的长宽）都生成一样的代码。模板里面的参数是可调的，这样如果输入代码超参数不同，它也会针对不同的超参数做一些对应的优化。所以对于XLA和Glow可观赏的只有HLO这一层，这也是比较务实的做法，因为第二阶段优化太难了。TVM，TC, Tiramisu, PlaidML使用了DSL，这使得它们能表示的运算范围更广，当然也直面了第二阶段优化的挑战，这也是这些编译器出彩的地方，我们在下一个要点里讨论第二阶段优化。对第一阶段优化来说，XLA做的相当全面了，可能是最全面的。

## 后端优化

2，剑宗还是气宗？在特定体系结构上自动生成最优代码，可以看出有俩套路，一个套路可以称之为剑宗，也好似一种自底向上的办法，即把专家手工优化代码的经验提炼出来，形成一些粗线条的rule, 这是TVM的路线，也就是来自Halide的思路，那些rule称为schedule，Halide的最重要贡献是提出了问题表示方法，但没有解决自动化这个问题，当然自动化问题很困难。还有一个套路可称之为气宗，寻求一种高层次的数学抽象来描述和求解代码生成问题，这个代表就是TC, Tiramisu, MLIR等依赖的Polyhedral（多面体模型， 仿射变换，并行处理）方法。

### 多面体模型

深度学习编译器和传统编译器技术很不相同，它只解决一类特定问题，不去解决控制流问题，基本上是解决多重循环优化，也就是稠密计算问题。Polyhedral method 是一种对多重循环程序的表示方法，问题表示出来之后，并不一定要借助isl求解。isl是什么？对一个比较复杂的cost model, 一个polyhedral 表示的计算，生成什么样的代码最优？这是一个离散解空间上的组合优化问题，有时可描述成整数线性规划，isl (integer set library)就是来求解这个问题的一个开源库。像TC， Tiramisu 用了isl, PlaidML和MLIR仅仅用了多面体表示却没有借助isl, 猜测：问题搜索空间太大时，isl 也不行。多面体方法只解决表示问题，不解决自动化问题。用多面体与否实际上和前端是否用DSL也有关，用Polyhedral 的都用DSL, 表明多面体的表达能力和能求解的问题范围非常广，用DSL但不用Polyhedral 也有，譬如Halide, TVM, Tiramisu 作者对Halide表达能力有所批评，需要注意的是， Halide和Tiramisu 作者们其实是同一位MIT教授的学生，这是自己对自己的批评。

@要术甲杰

未来真正的方案可能是剑宗和气宗结合，bottom up 经验有助于缩小搜索空间。PlaidML和Tiramisu 这方面做得很好。


## CPU和GPU的设计考虑

软件（编译器）通过对卷积操作的变换来减少不必要的MACs，例如前面提到的卷积近似计算通过编译时的优化PASS对循环执行进行调度（循环展开、循环tile等等）编译器优化减少load,store指令，尽量用寄存器操作2. 硬件提高PEs处理每个MAC的时间.提高并行度（使用更多并行的PEs，例如PE阵列）.提高PE的利用率. 意思是尽量使得PE忙碌，一个方法是增加片上缓存大小，提高访存总线宽度等等入手，这些都旨在减少访存的瓶颈，PEs可以更快拿到数，减少等待时间；另一种方式是让数据尽可能在数据通路中多停留，这样在访存能力不变的情况下，就可以用流水线的形式让PE尽可能有事做，后面讲解的TPU所用的脉动阵列，和流行的 DtaFlow数据流架构都是类似方式，当然在GPU的设计上也可以融入上述通用思想.

DNN加速器在体系结构设计所关注的几个点：

数据的读写也就是访存，以及计算。

尤其是访存问题，因为通常PE数目很容易堆上去，并行度上去后，数据读写往往会成为瓶颈，也就是常说的撞到**“内存墙”**，内存访问在我们的机器上是有层级的，寄存器，cache很快，但是到了内存取数就很慢，数据在cache miss后代价会很大，因此如果没有很好的访存优化，那么对于大数据的深度学习，加速比是很难做上去的。

从DNN加速器的经典实现 DianNao系列、TPU、Eyeriss来看（后面会讲到),这些加速器都十分关注访存方面的设计，后续有时间就其一些设计细节展开。

不过大体上优化访存的方式有

0。研究新的内存制作工艺：这个没那么没那么容易其实

1。从更下级cache取数：一般CPU从L1 cache取数，但是L1 cache能装的数比较少，我看到有些加速器从L2 cache取数，因为L2 更大一些，取数 miss几率小，不过这么做要解决一下cache一致性问题（Hwacha）。

2。增大加速器的片上buffer：好理解，如果我加速器的buffer大的可以装进更多权重，那么计算部件PE要消费数据的时候取数从片上buffer取，那肯定比从内存取数要快。

3。数据重用：如果这个PE产生的结果下一次计算会被接着用上，那么可以让这个数据留在PE中，等待下次复用，这样就减少了数据被store回去、下一次又load回来的开销。

从另外一个角度看，第二种方式本质上也是在利用数据的复用，只不过把数放在片上buffer来复用，复用的粒度不同。


### 数据复用

前面说道，处理访存问题是DNN加速器设计的一个关键，一个方法就是发掘卷积操作中的数据复用性。

卷积神经网络三个角度复用，

一是复用权重；

二是复用激活（也就是feature map）；

三是复用两者。

不同的复用策略，可以引申出不同设计架构和取数策略：

https://www.zhihu.com/search?q=glow%20%20tvm&utm_content=search_history&type=content


### 稀疏数据计算

为什么稀疏是一个对硬件设计友好的特性呢？首先，减少不必要的乘法操作。0乘上任何数都是0，那么如果有0出现的话可以不使能乘法单元去计算，直接输出结果0，这样可以节省运算产生的功耗；乘法的操作数有0的话，不必费力去内存取0，直接有个存0寄存器输出0给乘法单元即可，这样就减少了数据搬移的开销。体系结构设计上可以利用剪枝的
思想，把很小的只裁剪成0，这样就增大数据稀疏性，这里说的剪枝是硬件实现的剪枝。


### 低精度

前面算法层面有讲过量化的加速作用，但是量化本身其实是一个软硬件结合的设计，怎么理解呢？你在程序上实现的量化算法，落到指令架构级别都需要硬件的支持才行。这里再加强一下，INT8低精度支持可以使得数据的搬移量减少，同时整形操作速度更加快从而起到加速效果，目前许多设备都开始支持低精度推理

### 压缩

同样，如果硬件支持压缩功能，那么也可以减少数据搬移，例如，Eyeriss在写回数据时候会进行压缩编码再写回，这样大大减少访存的时间。


各类机器学习的行为模式很大不同（相比各类DNN而言），但是从更细粒度，也就是在数据重用、计算操作（乘加、计数比较..）角度上是可以分几类的，针对这一点设计如下的结构：

HotBuf和ColdBuf 就是在分析几类算法的数据重用行性后设计的，有的数据重用距离小，有的数据重用距离大，所以将buffer根据重用距离进行分离。

数据通路根据总结了几类算法出现的高频运算操作（有比较操作、计数、乘加（乘法+加法树组合）、中间结果缓存、非线性运算）做成流水线，对于某种特定的类型，如果在某级不需要该运算，可以直接bypass.PuDianNao论文的详细分析可以参考：中科院说的深度学习指令集diannaoyu到底是什么?大佬讲的蛮好的。

另外，如果想细究一下加速器一些具体的设计细节，可以看看ShiDianNao，这篇论文没有太多架构创新，但是在加速器设计细节上讲的应该是所有DianNao系列论文中最细致的了



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
