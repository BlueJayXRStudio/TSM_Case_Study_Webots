# import py_trees

# class RepeatUntilSuccess(py_trees.decorators.Decorator):
#     def __init__(self, name="RepeatUntilSuccess", child=None):
#         super().__init__(name=name, child=child)

#     def update(self):
#         if self.decorated.status != py_trees.common.Status.SUCCESS:
#             # Tick the child again
#             self.decorated.tick_once()
#             if self.decorated.status == py_trees.common.Status.SUCCESS:
#                 return py_trees.common.Status.SUCCESS
#             else:
#                 return py_trees.common.Status.RUNNING
#         else:
#             return py_trees.common.Status.SUCCESS