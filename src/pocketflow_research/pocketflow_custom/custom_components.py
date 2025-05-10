import copy
import warnings
from pocketflow import Node as BasePocketNode, Flow as BasePocketFlow, BatchNode as BasePocketBatchNode

# Custom Node and Flow classes for PocketFlow 
# track node names and flow names for better debugging and logging.

class CustomNode(BasePocketNode):
    def __init__(self, name: str = "", max_retries=1, wait=0):
        super().__init__(max_retries=max_retries, wait=wait)
        self.node_name = name if name else self.__class__.__name__

class CustomBatchNode(BasePocketBatchNode):
    def __init__(self, name: str = "", max_retries=1, wait=0):
        super().__init__(max_retries=max_retries, wait=wait)
        self.node_name = name if name else self.__class__.__name__

class CustomFlow(BasePocketFlow):
    def __init__(self, start=None, name: str = ""):
        super().__init__(start=start)
        self.flow_name = name if name else self.__class__.__name__
        if start and not hasattr(start, 'node_name'):
            warnings.warn(
                f"Start node {start.__class__.__name__} in Flow {self.flow_name} "
                "is not a CustomNode or CustomBatchNode. Node name tracking might not work as expected for it."
            )

    def _orch(self, shared, params=None):
        curr = copy.copy(self.start_node)
        p = params or {**self.params}
        last_action = None

        if not curr:
            warnings.warn(f"Flow '{self.flow_name}' has no start node.")
            return None

        while curr:
            curr.set_params(p)
            if hasattr(curr, 'node_name'):
                shared["current_node"] = curr.node_name
                # print(f"Executing Node: {curr.node_name}") # For debugging
            else:
                shared["current_node"] = curr.__class__.__name__
                warnings.warn(
                    f"Node {curr.__class__.__name__} in Flow {self.flow_name} "
                    "does not have a 'node_name' attribute. Using class name."
                )
                # print(f"Executing Node (class name): {curr.__class__.__name__}") # For debugging

            last_action = curr._run(shared)
            
            if hasattr(curr, 'node_name'):
                shared["current_node"] = f"{curr.node_name} (Completed)"

            next_node_candidate = self.get_next_node(curr, last_action)
        
            curr = copy.copy(next_node_candidate) if next_node_candidate else None
            
        return last_action
    

# TODO: Create custom versions for AsyncNode, AsyncFlow, etc., if used,
# ensuring they also correctly set current_node_name and flow_name.
