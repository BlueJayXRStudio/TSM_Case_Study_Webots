# TSM_Case_Study_Webots
TSM Case Study in Webots  

## Development Environment: 
**Webots version: R2023b**  
**Python version: 3.11.4**

## Setup:
In root directory, please run:

> git submodule update --init --recursive  

to initialize submodules (PyInfiniteTree)

or: 

> git clone https://github.com/BlueJayXRStudio/PyInfiniteTree.git  

into "./controllers/Controller_InfiniteTree/PyInfiniteTree"

## Summary:

Although higher-level planning algorithms such as A* and RRT, combined with obstacle detection and local replanning, can solve the majority of robotic navigation tasks, real-world environments continue to present challenges. Robots with articulated or bulky mechanical components often become stuck on environmental features like chairs, tables, and walls. These issues are especially common during the early mapping phase, when a robot is first learning the space and operating with a preliminary set of waypoints. If we assume that the robot will eventually rely on waypoint navigation for its long-term mobility, it becomes imperative to equip it with robust mechanisms for recovering from seemingly minor, yet impassable, states.

<img src="design_docs/readme_resources/WPNavAsFSM.png" alt="WPNavAsFSM" width="300"/>

<img src="design_docs/readme_resources/WPNavAsDynamicBT.png" alt="WPNavAsFSM" width="500"/>

While this was the original motivation for the work, it led to a deeper realization: even the task of moving to a single waypoint is not a primitive action, but a complex behavior. In practice, it involves managing orientation, retrying after failure, and selecting movement actions based on changing state. This is not well-modeled by a classical FSM, nor by traditional Behavior Tree implementations, which assume a static, pre-wired structure of nodes.

Colledanchise and Ögren address this limitation with a new formalization of Behavior Trees as recursive, functional compositions over system state and return status. This reformulation enables behaviors to be treated as composable, modular functions capable, in principle, of invoking subtasks and responding dynamically to their outcomes. However, this formalism is itself a departure from earlier BT models, and most practical implementations do not take it to its logical conclusion.

A modified use of py_trees can simulate this dynamic recursion by manually tracking and ticking subtasks, but doing so requires imperative logic and external state management (e.g., holding a reference to self.current_subtree and checking its status explicitly). While this demonstrates the expressive power enabled by the formalism, it remains structurally tied to the traditional BT engine model and does not naturally reflect the recursive, call-stack-based flow suggested by the theory.

The Task Stack Machine (TSM) fills this gap by providing a runtime execution model that directly matches the recursive semantics of the formalism. Each task in TSM is a first-class execution unit that can dynamically call subtasks, propagate return statuses, and build control flow at runtime. Behaviors like MoveToWaypoint are no longer statically composed trees; they become dynamically constructed processes made up of many nested actions, retries, and reactive adjustments.

This approach finds further theoretical support in the work of Florez-Puga et al., whose Dynamic Behavior Trees explicitly acknowledge the need for runtime behavior generation in response to environmental context. Together, these frameworks reinforce the idea that behaviors must be both composable and dynamically constructed in order to handle the demands of real-world autonomy.

Building on this foundation, TSM does not merely agree with the semantic direction of modern BT theory—it operationalizes it. The diagrammatic structure introduced in this case study is not only a practical architecture for robust navigation; it also represents a novel and faithful implementation of the recursive, compositional execution model that Colledanchise and Ögren and Florez-Puga et al. conceptually endorse but stop short of fully implementing.


## References:

[1] Michele Colledanchise and Petter Ögren. Behavior trees in robotics and AI: An introduction. CRC Press, 2018.  
[2] Gonzalo Flórez-Puga, Marco Gomez-Martin, Belen Diaz-Agudo, and Pedro Gonzalez-Calero. Dynamic expansion of behaviour trees. In Proceedings of the AAAI conference on artificial intelligence and interactive digital entertainment, volume 4, pages 36–41, 2008.