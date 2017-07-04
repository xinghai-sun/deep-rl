# A Brief Survey of Goal-driven Language Acquisition

## Survey Papar List

- [Listen, Interact and Talk: Learning to Speak via Interaction](https://arxiv.org/pdf/1705.09906.pdf)
- [A Deep Compositional Framework for Human-like Language Acquisition in Virtual Environment](https://arxiv.org/pdf/1703.09831.pdf)
- [Grounded Language Learning in a Simulated 3D World](https://arxiv.org/pdf/1706.06551.pdf)
- [Emergence of Grounded Compositional Language in Multi-Agent Populations](https://arxiv.org/pdf/1703.04908.pdf)
- [A Paradigm for Situated and Goal-Driven Language Learning](https://arxiv.org/pdf/1610.03585.pdf)
- [Multi-Agent Cooperation and the Emergence of (Natural) Language](https://arxiv.org/pdf/1612.07182.pdf)
- [Learning to Communicate with Deep Multi-Agent Reinforcement Learning](https://arxiv.org/pdf/1605.06676.pdf)
- [Learning Multiagent Communication with Backpropagation](https://arxiv.org/pdf/1605.07736.pdf)
- [Beating Atari with Natural Language Guided Reinforcement Learning](https://arxiv.org/pdf/1704.05539.pdf)
- [Dialog-based Language Learning](https://arxiv.org/pdf/1604.06045.pdf)


## Overview

### 自然语言形成的原因和条件

智能体（agents），例如人类，通常生活在群体中。在群体中，多智能体往往需要协同行动，共同完成某项任务（例如人类的共同狩猎，无人车协同行驶等）。由于各智能体只可观测或了解到环境中的部分信息（partial observability），为了更高效地完成协同任务，彼此的信息交换（不仅指环境信息，还包括行动信息）便成为一个必然需求，即通信（communication）。

通信需要信息的发送者和接收者遵循同一个约定的通信协议（message protocal），以便信息得以编码、在有限带宽的信道（communication channel）中传播、解码复原，达到信息传递的目的。不同的智能体在进化中往往选择了不同的通信协议和通信信道，例如蜜蜂，通过将信息编码为 “舞蹈”模式，以视觉的信道以传播。而人类则选择离散的符号序列来构造我们的通信协议，我们称之为自然语言（natural language)。在这样的一个通信机制的存在下，智能体得以交换信息、交互和协调行为，提升协同工作的效率。

语言形成的另外一个重要因素，是通信信道的有限带宽（例如离散符号序列的带宽取决于离散符号数）。在无限带宽情形下，语言不一定会产生。例如人的左右半脑，尽管各有分工，但通过胼胝体中的海量神经元紧密相连、协同工作，其近乎无限大的通信带宽使得左右半脑直接沟通，而不需要通过语言（见xxx)。

总结来说，语言，是多智能体在面临部分可观测环境时，为了共同完成某个任务并提升完成的效率，而演化出的通信协议。即语言形成的重要因素有：

- 多智能体环境
- 各智能体面临环境的部分可观测性
- 共同完成任务的需求
- 有限带宽通信信道

同时，复杂语言的习得，是人类认知的关键部分，是人工智能需要攻克的关键里程碑。

### 模拟环境下的语言产生和习得

根据上述，语言的本质特性除了通信协议的**统计结构**外，语言的**功能性**，即 “语言是服务于任务的完成”，是语言形成和习得的关键因素。然而近年来的自然语言研究主要集中在利用监督（或半监督）方法在**静态**标注语料上挖掘自然语言的**统计结构信息**（如统计语言模型），忽视语言的**功能性**特点。利用语言的**功能性**特定，将语言的习得根植于具体的任务驱动中，是 NLP 新的研究路径。

同时，人工智能体，在面临多智能体和协同任务环境时，同样存在上述的 “语言” 需求。研究多智能体在模拟的交互环境下是如何演化出类人的语言，有助于研究类人的认知；同时，当人类作为该多智能体环境中的一员时，我们希望 AI 不仅仅能进化出自有语言，同时可习得人的自然语言，即人类可通过自然语言完成和机器的沟通（如指令），和机器协同完成任务，是人类的一大梦想之一。

以上论文，即在这样的背景下，尝试在模拟的多智能体交互环境中，利用语言的功能性，以任务驱动的方式，研究语言的产生（language emergence）或自然语言的习得（language grounding / acquisition)，探讨机器与机器、机器与人的沟通。这些论文或方法，根据以下的侧重而有不同，

- 研究目的：Language Emergence Vs. Language Grounding。即产生新的语言 Vs. 习得现有人类语言。
- 通信对象： Machine-to-machine Vs. Machine-to-human Communication。即Without or with human in the loop。
- 通信方式：Broadcast Vs. One-one Communcaiton。即是以广播形式还是一对一形式通信。
- 通信信道：Discreted Channel Vs. Continuous Channel。即信道是离散信道还是连续信道。
- 研究侧重：侧重提升通信效率（关注return的提升) Vs. 侧重分析通信内容是否和人类语言相似。
- 交互类型：Teacher-student paradim Vs. Two-player paradim Vs. Multiple-player paradim.
- 模拟环境：不同的虚拟任务、不同的动作空间、不同的状态空间等。


## Notes on Paper Reading

以下不赘述论文细节，仅简要整理阅读笔记和思考。

### Natural Language Grounding

本节关注并比较以下四篇论文：

- [1] [Listen, Interact and Talk: Learning to Speak via Interaction](https://arxiv.org/pdf/1705.09906.pdf)
- [2] [A Deep Compositional Framework for Human-like Language Acquisition in Virtual Environment](https://arxiv.org/pdf/1703.09831.pdf)
- [3] [Grounded Language Learning in a Simulated 3D World](https://arxiv.org/pdf/1706.06551.pdf)
- [9] [Beating Atari with Natural Language Guided Reinforcement Learning](https://arxiv.org/pdf/1704.05539.pdf)

总结如下：

- 这四篇文章探讨如何在任务驱动的环境下，让机器理解人的自然语言指令（natural language grounding）。

- \[[1](#references)\] 和 \[[2](#references)\] 同出于百度 IDL 美研，且同作者。均模拟婴儿学习的环境，构造简单的带奖赏的视觉交互式任务，通过强化学习，训练 agent 在视觉环境中，习得根据自然语言指令 speak\[[1](#references)\] 和 navigate\[[2](#references)\] 的能力。二者均关注 natural language grouding in vision，前者侧重 “听说”，后者侧重 “听做”。

- \[[3](#references)\] 和 \[[2](#references)\] 的研究目的类似, 区别在于：\[[2](#references)\] 构造的是 2-D 环境导航任务，\[[3](#references)\] 则考虑更复杂的 3-D 情形。2-D 环境完全可观测，3-D 环境仅部分可观测 (需exploration）。 2-D 环境可借助 attention map 来描述导航目标，模型设计更 specific；而 3-D 环境则无法借助这样的中间媒介，模型设计更 general。

- \[[9](#references)\] 寻求和 \[[3](#references)\] \[[2](#references)\] 类似的自然语言指令导航，但以 Atari 的 Montezuma's Revenge 游戏为环境，试图通过自然语言定义 agent 的 subgoals. Montezuma's Revenge 游戏是典型的需要复杂的层次（hierachical）决策的场景，不过 \[[9](#references)\] 并未考虑自动的 macro 决策，而是尝试通过自然语言指令，给出人为指定的 macro subgoal。

- 上述 \[[1](#references)\]\[[2](#references)\]\[[3](#references)\]\[[9](#references)\] 均围绕 “让机器听懂自然语言指令” 这一目的。不同在于任务的设定：\[[1](#references)\] 为 “听，说”，\[[2](#references)\]\[[9](#references)\] 为 “听，2-D导航”， \[[3](#references)\] 为 “听，3-D导航” （注：这里的 “听” 和语音无关）。方法的整体框架有较大类似之处：均首先分别对视觉状态和指令状态进行编码（visual encoder & language encoder), 并将处理后的信号（可能额外叠加 recurrent 结构以弥补环境的部分可观测性）当做 RMDP 状态构建 policy function，最终以 actor-critic 方式求解 RMDP 问题。

- \[[1](#references)\]\[[2](#references)\]\[[3](#references)\]\[[9](#references)\] 均采用任务驱动的方法，实现 language grouding。传统的 NLP 更多地利用监督学习方法，在海量静态标注语料中，学习语言的统计结构信息。其信息局限于语言环境内部，忽视了语言作为工具，是外部世界的纽带。局限于语言环境内部的静态统计信息挖掘，是无法让 agent 掌握语言和环境关系。

- \[[1](#references)\]\[[2](#references)\]\[[3](#references)\]\[[9](#references)\] 的另一大相似之处在于，均采用辅助的监督或者无监督 task 来改善 RL 的收敛。\[[1](#references)\] 采用基于自然语言指令（含语言反馈）的语言模型建模；\[[2](#references)\] 采用 Visual Question Answer 监督任务；\[[3](#references)\] 采用无监督的下一帧预测任务和视觉预测语言指令任务；\[[9](#references)\] 采用有监督的 reward 预测任务。注意辅助的监督或无监督任务需和 policy network 在部分参数上共享，类似 multiple task learning，以共享统计强度，达到 “辅助” 的作用。例如 \[[1](#references)\] 中 language encoder 的 RNN 语言模型和 policy 中的 RNN 模型参数共享；\[[2](#references)\] 中的classification softmax weight 和 language word embedding 的参数共享；\[[3](#references)\]\[[9](#references)\] 中 policy network 的 visual encoder 和 language encoder 均和无监督或有监督任务模型共享参数。这样的辅助监督和无监督任务的存在非常必要，因环境的稀疏 reward 和巨大的策略空间，使得单纯的强化学习常常会陷于较差的策略空间而反复搜索，无法获得正向 reward 以实现 reinforced improvement。\[[1](#references)\] 以直观的方式解释这一现象，即婴儿往往同时拥有模仿学习（辅助的监督任务）和强化学习，并举鹦鹉学舌和听障人士的例子说明二者缺一不可。

- \[[1](#references)\]\[[2](#references)\]\[[3](#references)\]\[[9](#references)\] 均通过实验，证明监督或非监督辅助任务使得 agent 具有 zero-shot 的学习能力。

- \[[1](#references)\] 中的 policy network 设计思路非常有意思。为了充分利用 “模仿学习” （即上述的辅助监督任务）的好处，policy network 利用和 language encoder 相同的 RNN （参数共享）作为 decoder 以产生 speak 动作，同时将策略的控制权后移到条件向量（即 RNN 的初始状态）中，即通过一个controller function 来改变该条件向量。在强化学习训练时，固定 RNN 部分，仅更新 controller 部分。该结构将语言内容的控制（通过强化学习来习得）和语言形式的概率结构的控制（通过监督学习来习得）进行有效剥离。如此一来，policy network 即回退为该 controller，而该 RNN decoder 又可以融合成为环境的一部分。既降低了 policy gradient 的优化代价，又使得语言模型本身不受强化学习过程的不良影响。是一个非常有意思的设计思路。


### Language Emergence

本节关注并比较以下五篇论文：

- [4] [Emergence of Grounded Compositional Language in Multi-Agent Populations](https://arxiv.org/pdf/1703.04908.pdf)
- [5] [A Paradigm for Situated and Goal-Driven Language Learning](https://arxiv.org/pdf/1610.03585.pdf)
- [6] [Multi-Agent Cooperation and the Emergence of (Natural) Language](https://arxiv.org/pdf/1612.07182.pdf)
- [7] [Learning to Communicate with Deep Multi-Agent Reinforcement Learning](https://arxiv.org/pdf/1605.06676.pdf)
- [8] [Learning Multiagent Communication with Backpropagation](https://arxiv.org/pdf/1605.07736.pdf)

总结如下：

- 这五篇文章均探讨在无人类参与情形下（without human in the loop)，多智能体在任务驱动的协作下，如何产生符号（序列）化的语言，并利用该语言提升协作的效率。同时探讨该语言和人类语言的相似之处 \[[6](#references)\]（例如是否包含 abstract concept语义）。此外，这一多智能体生成语言的场景，在有人类参与（with human in the loop) 并使用自然语言的情形下，很容易拓展出多智能体对人类自然语言的学习能力。

- \[[4](#references)\] 和 \[[5](#references)\] 均出自 Igor Mordatch。\[[4](#references)\] 通过构造以下特点的环境，并建模任务驱动的语言产生。
    - 部分可观测性：每个 agent 均被指派一个仅有自己知道的 goal，该 goal 和自己无关，需要通过信息交流，通知其他某目标 agent 来完成特定任务。这一任务指定方式，使得每个智能体可观测的状态具有部分可观测性（其他 agents 的 goal 的不可观测性），于是有信息沟通的需求。
    - 离散的通信信道：人类语言的信道通常为离散符号序列，\[[4](#references)\] 构造类似的离散序列信道。同时为简化问题，该信道为 broadcast 形式，即任何 agent 均可听到其他 agent 发出的信息，且信息彼此不干扰。
    - 将语言的发出也视作是一个动作，即 agent 即有 $u(s)$ 的移动动作，又有 $c(s)$ 的通信动作，$u$ 将改变环境的状态，而 $c$ 不改变状态，只影响其他 agent 的动作选择。每个时间步，agent 只可选择一个抽象的符号，发送至公共信道中。
    - 通信信道的有限带宽：通过soft（加权）方式限制通信信道的带宽（即限制可用于通信的抽象符号的数量，即 vocabulary size），可演化出具有组合特性（compositionality）的符号序列。显然，单时间步的通信带宽有限时，智能体会选择以多步组合的方式，发出信息，这和人类语言特点是一致的。
    - Agent 的数量可变。于是要求每个智能体需要以相同的方式处理每个其他 agent 的 message 和 state，尤其在 broadcast 的通信方式下。
    - Group reward 假设，即任何一个 agent 的 goal 的完成时，所有agent均获得相同的 group-level reward。即假设完全协作状态。

    在上述构造的环境中，agent 通过观察:(1)自身和所有其他智能体的状态，(2)所有其他智能体发出的 message, (3)自身的goal，做出以下两种动作决策: 移动动作 $u$ (移动方向，眼神指向等） 和 交流动作 $c$ （选择一个抽象字符发出）。如此，构成一个标准的 MDP 过程。求解该 MDP，智能体演化出特殊的沟通方式。有意思的是，智能体不仅仅能通过 $c$ 来发出通信信息，同时 “眼神指向”（文中的 gaze direction) 等也被利用作为通信媒介，类似人的肢体语言和表情。

- \[[5](#references)\] 将上述 \[[4](#references)\] 描述的任务驱动的语言学习 paradigm 做了更多阐述。同时提出了当 involve human in the loop, 并且人坚持说自然语言，可以让机器在特定感知环境下习得人的自然语言（natural language grounding），而不仅仅是发明新的不被人类理解的语言。

- \[[6](#references)\] focus 在探究机器自主发明的语言是否可被人类理解（interpretable），及它和人类的自然语言的相似之处，例如，是否具有 high-level abstract 语义。和 \[[4](#references)\] 不同在于，为了探讨语义，\[[5](#references)\] 构造一个语义相关的交流任务：两个agents，sender agent 和 receiver agent，给定两张随机图片（不同语义类别），sender agent 需要选定其中一张图片并选择一个抽象字符作为信息发送给 receiver，receiver 根据该字符猜测 sender 选定的图片是哪一张。这里仅有两个 agents，并且通信的信道仅限于一个离散字符（非序列），同时游戏为单步的。\[[6](#references)\] 通过分析 agent 选择的字符和图片语义的相关性，证明在 agents 的协作沟通过程中，智能体是有可能产生 high-level 的具有抽象语义的符号化语言，来表达并沟通他们对 high-level 视觉信息的理解。

- \[[7](#references)\] 的创新在于考虑可微分的信息通道。上述所有均采用离散不可微分的信道，即训练时网络的梯度无法在不同的 agent 之间传播。然而，既然不同的 agent 之间并非相互独立（communication、action 等均破坏独立性），在训练优化时，对网络参数的梯度就应该考虑不同 agent 之间的策略依赖性。\[[7](#references)\] 考虑在训练时，以 sigmoid 连续信道近似在实际执行时采用的离散信道，使得训练过程中梯度可以通过通信模块得以在不同的 agent 之间传递。起到显著加速强化训练的目的。当然在执行阶段（非训练阶段），agents 仍然以不可微的离散信道进行通信，此时没有梯度传递需求。

- \[[7](#references)\] 同时提出 agents 之间的通信可以分为两种类型，即学习时通信和执行时通信。实际中大量任务满足 Centralized Learning，Decentralized Execution 要求，即训练时多 agent 可以有海量带宽的连续信道进行通信，看起来就像仅有一个智能体在学习(中心化的学习），而执行时，各个 agent 又各有各的状态，彼此间依赖有限带宽的离散信道进行通信（去中心化的执行）。在这种设定下，训练时的我们就可以利用更大的通信带宽来更好得协调并加速多智能体的共同学习。例如上述的多智能体之间的梯度传递即利用该学习时通信。同时多智能体 share 相同 policy，也可以认为是学习时通信的一个简单例子。

- \[[8](#references)\] 考虑了连续通信信道情形（指执行时）。即不同 agent 之间可通过类似神经网络（nerual network）连接的方式，在各自的 policy network 之间形成连续值的信息通路。当然这种情形下，语言是无法产生的。但当 agent 间允许有有近乎无限带宽的通信存在时（例如即时战略游戏中，同盟 AI agents 间可通过内存来交换信息，当然这种设定有时是作弊情形），我们可以考虑 \[[8](#references)\] 的模型设计，为各 agents 的 policy network 之间添加特定的 communication connections。\[[8](#references)\] 将这种特殊的含通信目的的连接网络称作 CommNet, 同时证明了在通信条件允许的情形下，这种连续通信信道要远优于选择离散信道。

- \[[7](#references)\] \[[8](#references)\] 通过实验说明了 multi-agent 在通信存在下，协作效率均显著提升。

- 总结来看，这五篇文章均探讨 multi-agent 在为提升协作任务效率的目的下，产生的信息交流，及在信息交流中可能产生的语言，以及这种机器发明的语言和人类进化获得的自然语言的相似性。其中，\[[8](#references)\] 探讨了连续信道情形下的交流，\[[7](#references)\] 探讨了离散信道情形下跨 agent 的梯度传播，\[[6](#references)\] 则重点探讨机器产生的符号语言和人的语言之间的相似性，\[[4](#references)\]\[[5](#references)\] 同样通过实验证明了多智能体协同任务驱动下的语言的产生及其对协同任务效率的提升。


## Summary

上述的研究可以看成三个脉络：

- 探讨更高效的多 agents 间的通信方式（注意 “高效” 是因为信道的带宽通常是有限的）。
- 探讨这一通信是否可能催生出的语言（探究语言的形成、语言对认知的影响）。
- 探讨当 human is involved in the loop 时，机器能否理解人的语言。

同时上述的探讨，均在任务驱动这一背景下，模拟可控的简单环境来进行的。

这些研究，不仅对 multi-agent cooperation 相关任务有帮助，同时对 NLP 领域也有着深远的影响。更进一步，不仅仅是语言，人类其他的认识能力也和任务驱动紧密相关，即各种认知都是在工具性和任务驱动的目的下产生的。这一思考方式，有助于我们将监督、无监督、强化学习等领域，将感知和认知，更紧密结合在一起，推动 AI 的发展。


## References
1. Zhang H, Yu H, Xu W. [Listen, Interact and Talk: Learning to Speak via Interaction](https://arxiv.org/pdf/1705.09906.pdf)[J]. arXiv preprint arXiv:1705.09906, 2017.
2. Yu H, Zhang H, Xu W. [A Deep Compositional Framework for Human-like Language Acquisition in Virtual Environment](https://arxiv.org/pdf/1703.09831.pdf)[J]. arXiv preprint arXiv:1703.09831, 2017.
3. Hermann K M, Hill F, Green S, et al. [Grounded Language Learning in a Simulated 3D World](https://arxiv.org/pdf/1706.06551.pdf)[J]. arXiv preprint arXiv:1706.06551, 2017.
4. Mordatch I, Abbeel P. [Emergence of Grounded Compositional Language in Multi-Agent Populations](https://arxiv.org/pdf/1703.04908.pdf)[J]. arXiv preprint arXiv:1703.04908, 2017.
5. Gauthier J, Mordatch I. [A paradigm for situated and goal-driven language learning](https://arxiv.org/pdf/1610.03585.pdf)[J]. arXiv preprint arXiv:1610.03585, 2016.
6. Lazaridou A, Peysakhovich A, Baroni M. [Multi-agent cooperation and the emergence of (natural) language](https://arxiv.org/pdf/1612.07182.pdf)[J]. arXiv preprint arXiv:1612.07182, 2016.
7. Foerster J, Assael Y M, de Freitas N, et al. [Learning to communicate with deep multi-agent reinforcement learning](https://arxiv.org/pdf/1605.07736.pdf)[C]//Advances in Neural Information Processing Systems. 2016: 2137-2145.
8. Sukhbaatar S, Fergus R. [Learning multiagent communication with backpropagation](https://arxiv.org/pdf/1605.07736.pdf)[C]//Advances in Neural Information Processing Systems. 2016: 2244-2252.
9. Kaplan R, Sauer C, Sosa A. [Beating Atari with Natural Language Guided Reinforcement Learning](https://arxiv.org/pdf/1704.05539.pdf)[J]. arXiv preprint arXiv:1704.05539, 2017.
10. Weston J E. [Dialog-based language learning](https://arxiv.org/pdf/1604.06045.pdf)[C]. Advances in Neural Information Processing Systems. 2016: 829-837.
