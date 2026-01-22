import numpy as np

class FOICPolicy:
    """
    Fixed Offset with Inventory Constraints
    """

    def __init__(self, offset_a=0, offset_b=0):
        self.offset_a = offset_a
        self.offset_b = offset_b

    def get_action(self, inventory):
        return np.array([self.offset_a, self.offset_b])
    
class GLFTPolicy:
    """
    Guéant-Lehalle-Fernandez-Tapia (GLFT) approximation
    """

    def __init__(self, gamma=0.001, kappa=100, A=0.1, sigma=0.01):
        self.gamma = gamma  # Risk aversion coefficient 
        self.kappa = kappa  # LOB shape parameter k 
        self.A = A          # Trading intensity A 
        self.sigma = sigma  # Standard deviation 

    def get_action(self, inventory):
        # optimal quote
        phi = np.sqrt(
            (self.sigma**2 * self.gamma) / (2 * self.kappa * self.A) * (1 + self.gamma / self.kappa)**(1 + self.kappa / self.gamma)
        )

        spread_term = (1 / self.gamma) * np.log(1 + self.gamma / self.kappa)

        # Calculate offsets relative to the Mid-price
        # ask_offset = Q_ask - Mid
        # bid_offset = Mid - Q_bid
        q = inventory
        ask_offset_mid = spread_term + ( (2 * q - 1) / 2 ) * phi
        bid_offset_mid = spread_term - ( (2 * q + 1) / 2 ) * phi
        
        return np.array([ask_offset_mid, bid_offset_mid])