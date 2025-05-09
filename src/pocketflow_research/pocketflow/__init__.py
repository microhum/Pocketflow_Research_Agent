"""
PocketFlow: A minimalist LLM framework for Agents, Task Decomposition, RAG, etc.
"""

class Node:
    """
    Base class for all nodes in a flow.
    """
    def __init__(self, max_retries=1, wait=0):
        self.params = {}
        self.max_retries = max_retries
        self.wait = wait
        self.cur_retry = 0
        self.transitions = {}
        
    def set_params(self, params):
        """Set parameters for this node."""
        self.params = params
        return self
        
    def prep(self, shared):
        """Prepare data from shared store for execution."""
        return None
        
    def exec(self, prep_res):
        """Execute the node's main logic."""
        return None
        
    def exec_fallback(self, prep_res, exc):
        """Handle exceptions from exec."""
        raise exc
        
    def post(self, shared, prep_res, exec_res):
        """Process results and update shared store."""
        return "default"
        
    def run(self, shared):
        """Run the node's full lifecycle."""
        prep_res = self.prep(shared)
        
        exec_res = None
        self.cur_retry = 0
        
        while self.cur_retry < self.max_retries:
            try:
                exec_res = self.exec(prep_res)
                break
            except Exception as e:
                if self.cur_retry == self.max_retries - 1:
                    # Last retry failed, use fallback
                    exec_res = self.exec_fallback(prep_res, e)
                    break
                
                import time
                time.sleep(self.wait)
                self.cur_retry += 1
        
        action = self.post(shared, prep_res, exec_res)
        return action if action is not None else "default"
        
    def __rshift__(self, other):
        """
        Operator overload for >>
        node_a >> node_b is equivalent to node_a - "default" >> node_b
        """
        self.transitions["default"] = other
        return other
        
    def __sub__(self, action):
        """
        Operator overload for -
        Used in node_a - "action" >> node_b
        """
        class _TransitionBuilder:
            def __init__(self, node, action):
                self.node = node
                self.action = action
                
            def __rshift__(self, other):
                self.node.transitions[self.action] = other
                return other
                
        return _TransitionBuilder(self, action)


class BatchNode(Node):
    """
    A node that processes items in batches.
    """
    def exec(self, prep_res):
        """
        Process each item in the batch.
        
        Args:
            prep_res: An iterable of items to process
            
        Returns:
            list: Results for each item
        """
        results = []
        for item in prep_res:
            result = super().exec(item)
            results.append(result)
        return results


class Flow(Node):
    """
    A flow that connects multiple nodes.
    """
    def __init__(self, start=None):
        super().__init__()
        self.start = start
        
    def run(self, shared):
        """
        Run the flow from the start node.
        
        Args:
            shared: The shared data store
            
        Returns:
            str: The final action
        """
        if not self.start:
            return "default"
            
        # Apply flow params to start node
        self.start.set_params(self.params)
        
        current = self.start
        while current:
            action = current.run(shared)
            
            # Find the next node based on the action
            current = current.transitions.get(action)
            
            # If we found a next node and it's not the end, apply params
            if current:
                current.set_params(self.params)
                
        return "default"


class BatchFlow(Flow):
    """
    A flow that runs multiple times with different parameters.
    """
    def prep(self, shared):
        """
        Return a list of parameter dictionaries.
        
        Args:
            shared: The shared data store
            
        Returns:
            list: A list of parameter dictionaries
        """
        return []
        
    def run(self, shared):
        """
        Run the flow multiple times with different parameters.
        
        Args:
            shared: The shared data store
            
        Returns:
            str: The final action
        """
        param_list = self.prep(shared)
        
        for params in param_list:
            # Merge flow params with batch params
            merged_params = {**self.params, **params}
            
            # Set merged params on the flow
            self.set_params(merged_params)
            
            # Run the flow with these params
            super().run(shared)
            
        return "default"
