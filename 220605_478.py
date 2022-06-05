# class Solution:

#     def __init__(self, radius: float, x_center: float, y_center: float):
#         self.x_center = x_center
#         self.y_center = y_center
#         self.r = radius


#     def randPoint(self) -> List[float]:
#         while True:
#             x, y = random.uniform(-self.r, self.r), random.uniform(-self.r, self.r)
#             if x * x + y * y <= self.r * self.r:
#                 return [self.x_center + x, self.y_center + y]


# Your Solution object will be instantiated and called as such:
# obj = Solution(radius, x_center, y_center)
# param_1 = obj.randPoint()

class Solution:

    def __init__(self, radius: float, x_center: float, y_center: float):
        self.x_center = x_center
        self.y_center = y_center
        self.radius = radius

    def randPoint(self) -> List[float]:
        r = sqrt(self.radius * self.radius * random.random())
        theta = random.random() * 2 * math.pi
        return [self.x_center + r * math.cos(theta), self.y_center + r * math.sin(theta)]

