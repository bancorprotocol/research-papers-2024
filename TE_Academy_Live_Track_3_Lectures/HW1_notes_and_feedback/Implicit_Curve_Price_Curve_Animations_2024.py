# %%
import numpy as np
from typing import Tuple, List, Dict
from tabulate import tabulate

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as path_effects
from matplotlib.animation import FuncAnimation

plt.style.use('dark_background')

# %%
"""                                                                                
                          ,(&@(,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,                  
                   ,%@@@@@@@@@@,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,.                 
              @@@@@@@@@@@@@@@@@&,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,.                
              @@@@@@@@@@@@@@@@@@/,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,                
              @@@@@@@@@@@@@@@@@@@,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,               
              @@@@@@@@@@@@@@@@@@@%,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,              
              @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@.              
              @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@.                
          (((((((((&@@@@@@@@@@@@@@@@@@@@@@@@@@@(,,,,,,,%@@@@@,                  
          (((((((((@@@@@@@@@@@@@@@@@@@@@@@@@@((((,,,,,,,#@@.                    
         ,((((((((#@@@@@@@@@@@/////////////((((((/,,,,,,,,                      
         *((((((((#@@@@@@@@@@@#,,,,,,,,,,,,/((((((/,,,,,,,,                     
         /((((((((#@@@@@@@@@@@@*,,,,,,,,,,,,(((((((*,,,,,,,,                    
         (((((((((%@@@@@@@@@@@@&,,,,,,,,,,,,/(((((((,,,,,,,,,.                  
        .(((((((((&@@@@@@@@@@@@@/,,,,,,,,,,,,((((((((,,,,,,,,,,                 
        *(((((((((@@@@@@@@@@@@@@@,,,,,,,,,,,,*((((((((,,,,,,,,,,                
        /((((((((#@@@@@@@@@@@@@@@@/,,,,,,,,,,,((((((((/,,,,,,,,,,.              
        (((((((((%@@@@@@@@@@@@@@@@@(,,,,,,,,,,*((((((((/,,,,,,,,,,,             
        (((((((((%@@@@@@@@@@@@@@@@@@%,,,,,,,,,,(((((((((*,,,,,,,,,,,            
       ,(((((((((&@@@@@@@@@@@@@@@@@@@&,,,,,,,,,*(((((((((*,,,,,,,,,,,.          
       ((((((((((@@@@@@@@@@@@@@@@@@@@@@*,,,,,,,,((((((((((,,,,,,,,,,,,,         
       ((((((((((@@@@@@@@@@@@@@@@@@@@@@@(,,,,,,,*((((((((((,,,,,,,,,,,,,        
       (((((((((#@@@@@@@@@@@@&#(((((((((/,,,,,,,,/((((((((((,,,,,,,,,,,,,       
       %@@@@@@@@@@@@@@@@@@((((((((((((((/,,,,,,,,*(((((((#&@@@@@@@@@@@@@.       
        &@@@@@@@@@@@@@@@@@@#((((((((((((*,,,,,,,,,/((((%@@@@@@@@@@@@@%          
         &@@@@@@@@@@@@@@@@@@%(((((((((((*,,,,,,,,,*(#@@@@@@@@@@@@@@*            
         /@@@@@@@@@@@@@@@@@@@%((((((((((*,,,,,,,,,,,,,,,,,,,,,,,,,              
         %@@@@@@@@@@@@@@@@@@@@&(((((((((*,,,,,,,,,,,,,,,,,,,,,,,,,,             
         @@@@@@@@@@@@@@@@@@@@@@@((((((((,,,,,,,,,,,,,,,,,,,,,,,,,,,,            
        ,@@@@@@@@@@@@@@@@@@@@@@@@#((((((,,,,,,,,,,,,,,,,,,,,,,,,,,,,,           
        #@@@@@@@@@@@@@@@@@@@@@@@@@#(((((,,,,,,,,,,,,,,,,,,,,,,,,,,,,,.          
        &@@@@@@@@@@@@@@@@@@@@@@@@@@%((((,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,          
        @@@@@@@@@@@@@@@@@@@@@@@@@@@@&(((,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,         
       (@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@((,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,        
       MB@RICHARDSON@BANCOR@(2024)@@@@@/,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,       
"""

# %% [markdown]
# # Homework One
# 
# ## Class Implementation

# %%
class HomeworkOne:
    """
    ### Analyzes and visualizes the trading dynamics and marginal rates between two tokens.

    ## Description:
    - This class simulates and visualizes the effects of a series of trades on a pair of tokens using specified initial coordinates.
    - It calculates the marginal rates, effective rates, and trade amounts, and plots the results.

    ## Attributes:
    | Attribute Name                 | Type                             | Description                                                                                                    |
    |--------------------------------|----------------------------------|----------------------------------------------------------------------------------------------------------------|
    | `token_pair`                   | `Dict[str, str]`                 | A dictionary mapping token symbols to their identifiers.                                                       |
    | `x_0`                          | `float`                          | Initial amount of the first token.                                                                             |
    | `y_0`                          | `float`                          | Initial amount of the second token.                                                                            |
    | `invariant`                    | `List[float]`                    | A list containing the product of initial amounts of the two tokens (x_0 * y_0).                                |
    | `trade_actions`                | `List[Tuple[str, float]]`        | A list of trade actions, each represented by a tuple containing trade type ('Dx' or 'Dy') and trade amount.    |
    | `trade_increments`             | `int`                            | The number of increments for each trade action.                                                                |
    | `coordinates`                  | `List[List[Tuple[float]]]`       | A list of lists containing coordinates (x, y) after each trade action.                                         |
    | `marginal_rate_dy_dx`          | `List[List[float]]`              | A list of lists containing marginal rates ∂y/∂x after each trade action.                                       |
    | `marginal_rate_dx_dy`          | `List[List[float]]`              | A list of lists containing marginal rates ∂x/∂y after each trade action.                                       |
    | `effecive_rate_delta_y_delta_x`| `List[List[float]]`              | A list of lists containing effective rates Δy/Δx after each trade action.                                      |
    | `effecive_rate_delta_x_delta_y`| `List[List[float]]`              | A list of lists containing effective rates Δx/Δy after each trade action.                                      |
    | `swap_amounts`                 | `List[List[Tuple[float]]]`       | A list of lists containing the amounts swapped (Δx, Δy) after each trade action.                               |
    | `swap_functions`               | `Dict[str, Dict[str, Callable]]` | A dictionary mapping trade types to their corresponding calculation functions.                                 |
    | `sigmoid_transition_factors`   | `np.ndarray`                     | An array of sigmoid transition factors used to smoothly transition trade values.                               |
    | `x_array`                      | `np.ndarray`                     | An array of x values used for plotting the implicit curve.                                                     |
    | `y_array`                      | `np.ndarray`                     | An array of y values corresponding to x values, used for plotting the implicit curve.                          |
    | `dy_dx_array`                  | `np.ndarray`                     | An array of ∂y/∂x values used for plotting the derivative curve.                                               |
    | `dx_dy_array`                  | `np.ndarray`                     | An array of ∂x/∂y values used for plotting the derivative curve.                                               |
    
    ## Methods:
    | Method Name                              | Description                                                               |
    |------------------------------------------|---------------------------------------------------------------------------|
    | `__init__`                               | Initializes the class with initial parameters for tokens and trades.      |
    | `calculate_Dy_from_x_y_Dx`               | Calculates Δy for a given Δx using x and y coordinates.                   |
    | `calculate_Dy_from_x_invariant_Dx`       | Calculates Δy for a given Δx using x coordinate and invariant.            |
    | `calculate_Dy_from_y_invariant_Dx`       | Calculates Δy for a given Δx using y coordinate and invariant.            |
    | `calculate_Dx_from_x_y_Dy`               | Calculates Δx for a given Δy using x and y coordinates.                   |
    | `calculate_Dx_from_y_invariant_Dy`       | Calculates Δx for a given Δy using y coordinate and invariant.            |
    | `calculate_Dx_from_x_invariant_Dy`       | Calculates Δx for a given Δy using x coordinate and invariant.            |
    | `calculate_sigmoid_transition_factors`   | Calculates sigmoid transition factors for smooth trade transitions.       |
    | `handle_output_discrepancies`            | Handles discrepancies in trade output calculations and logs the process.  |
    | `calculate_trade`                        | Calculates the trade output for a given trade input and increment.        |
    | `make_next_row`                          | Prepares the next row for storing trade results.                          |
    | `process_trades`                         | Processes all trades specified in the trade actions.                      |
    | `find_min_max_x`                         | Finds the minimum and maximum x coordinates for plotting.                 |
    | `calculate_x_array_from_bounds`          | Calculates an array of x values for plotting based on bounds.             |
    | `calculate_y_array_from_x_array`         | Calculates an array of y values corresponding to x values.                |
    | `calculate_derivative_arrays`            | Calculates arrays of derivatives ∂y/∂x and ∂x/∂y.                         |
    | `draw_arrows`                            | Draws arrows between two points on a plot.                                |
    | `setup_plot`                             | Sets up the plot layout and axes.                                         |
    | `plot_implicit_curve`                    | Plots the implicit curve on a given axes.                                 |
    | `plot_trade_points_and_arrows`           | Plots trade points and arrows between them on a given axes.               |
    | `plot_derivative_curves_and_arrows`      | Plots derivative curves and arrows on given axes.                         |
    | `plot_derivative_y_x`                    | Plots ∂y/∂x curve on a given axes.                                        |
    | `plot_derivative_x_y`                    | Plots ∂x/∂y curve on a given axes.                                        |
    | `configure_axes`                         | Configures axes appearance for the plot.                                  |
    | `draw_crosshairs`                        | Draws crosshairs at a specified point on the plot.                        |
    | `write_text_summary`                     | Writes a text summary of the trade details on a given axes.               |
    | `get_plot_values`                        | Retrieves plot values for a specific trade and increment.                 |
    | `plot_implicit_and_price_curves`         | Plots implicit and price curves for a specific trade and increment.       |
    """


    def __init__(
        self,
        token_pair: Dict[str, str] = {'x' : 'x', 'y' : 'y'},
        x_0: float = 1234,
        y_0: float = 5678,
        trade_actions: List[Tuple[str, float]] = [('Dx', +100), 
                                                  ('Dy', +321), 
                                                  ('Dx', -123), 
                                                  ('Dy', -321),
                                                  ('Dx', +150), 
                                                  ('Dy', +500), 
                                                  ('Dx', -150), 
                                                  ('Dy', -600)],
        trade_increments: int = 50,
    ):
        """
        ### Initializes the HomeworkOne class with initial parameters for tokens and trades.

        ## Parameters:
        | Parameter Name     | Type                      | Description                                                                                                 |
        |--------------------|---------------------------|-------------------------------------------------------------------------------------------------------------|
        | `token_pair`       | `Dict[str, str]`          | A dictionary mapping token symbols to their identifiers.                                                    |
        | `x_0`              | `float`                   | Initial amount of the first token.                                                                          |
        | `y_0`              | `float`                   | Initial amount of the second token.                                                                         |
        | `trade_actions`    | `List[Tuple[str, float]]` | A list of trade actions, each represented by a tuple containing trade type ('Dx' or 'Dy') and trade amount. |
        | `trade_increments` | `int`                     | The number of increments for each trade action.                                                             |

        ## Functionality:
        - Sets up initial coordinates, calculates invariants, initializes swap functions, and processes trades.
        - Initializes arrays for coordinates, marginal rates, effective rates, and swap amounts.
        - Calculates sigmoid transition factors (allows for smooth animations), x array, y array, and derivative arrays for plotting.
        """
        self.token_pair = token_pair
        self.x_0 = x_0
        self.y_0 = y_0
        self.invariant = [x_0*y_0]
        self.trade_actions = trade_actions
        self.trade_increments = trade_increments
        self.coordinates = [[(x_0, y_0), []]]
        self.marginal_rate_dy_dx = [[(-y_0/x_0), []]] 
        self.marginal_rate_dx_dy = [[(-x_0/y_0), []]]
        self.effecive_rate_delta_y_delta_x = [[]]
        self.effecive_rate_delta_x_delta_y = [[]]
        self.swap_amounts = [[]]
        self.swap_functions = {
            'Dy_from_Dx' : {'x_and_y' : self.calculate_Dy_from_x_y_Dx,
                    'x_and_invariant' : self.calculate_Dy_from_x_invariant_Dx,
                    'y_and_invariant' : self.calculate_Dy_from_y_invariant_Dx},
            'Dx_from_Dy' : {'x_and_y' : self.calculate_Dx_from_x_y_Dy,
                    'x_and_invariant' : self.calculate_Dx_from_x_invariant_Dy,
                    'y_and_invariant' : self.calculate_Dx_from_y_invariant_Dy}
                    }
        self.sigmoid_transition_factors = self.calculate_sigmoid_transition_factors()
        self.process_trades()
        self.x_array = self.calculate_x_array_from_bounds()
        self.y_array = self.calculate_y_array_from_x_array()
        self.dy_dx_array, self.dx_dy_array = self.calculate_derivative_arrays()

    
    def calculate_Dy_from_x_y_Dx(
            self,
            x_coordinate: float, 
            y_coordinate: float,
            Dx: float,
    ) -> float:
        """
        ### Calculates Δy for a given Δx using x and y coordinates.

        ## Parameters:
        | Parameter Name | Type    | Description            |
        |----------------|---------|------------------------|
        | `x_coordinate` | `float` | The x coordinate.      |
        | `y_coordinate` | `float` | The y coordinate.      |
        | `Dx`           | `float` | The change in x (Δx).  |

        ## Returns:
        | Return Name | Type    | Description               |
        |-------------|---------|---------------------------|
        | `Dy`        | `float` | The change in y (Δy).     |

        ## Notes:
        - Uses the formula: `Dy = - (Dx * y) / (x + Dx)`.
        """
        numerator = Dx * y_coordinate
        denominator = x_coordinate + Dx
        return - numerator / denominator
    
    
    def calculate_Dy_from_x_invariant_Dx(
            self,
            x_coordinate: float, 
            invariant: float, 
            Dx: float,
    ) -> float:
        """
        ### Calculates Δy for a given Δx using x coordinate and invariant.

        ## Parameters:
        | Parameter Name | Type    | Description            |
        |----------------|---------|------------------------|
        | `x_coordinate` | `float` | The x coordinate.      |
        | `invariant`    | `float` | The invariant (x * y). |
        | `Dx`           | `float` | The change in x (Δx).  |

        ## Returns:
        | Return Name | Type    | Description               |
        |-------------|---------|---------------------------|
        | `Dy`        | `float` | The change in y (Δy).     |

        ## Notes:
        - Uses the formula: `Dy = - (invariant * Dx) / (x * (x + Dx))`.
        """
        numerator = invariant * Dx
        denominator = x_coordinate * (x_coordinate + Dx)
        return - numerator / denominator 
    
    
    def calculate_Dy_from_y_invariant_Dx(
            self,
            y_coordinate: float, 
            invariant: float, 
            Dx: float,
    ) -> float:
        """
        ### Calculates Δy for a given Δx using y coordinate and invariant.

        ## Parameters:
        | Parameter Name | Type    | Description               |
        |----------------|---------|---------------------------|
        | `y_coordinate` | `float` | The y coordinate.         |
        | `invariant`    | `float` | The invariant (x * y).    |
        | `Dx`           | `float` | The change in x (Δx).     |

        ## Returns:
        | Return Name | Type    | Description                  |
        |-------------|---------|------------------------------|
        | `Dy`        | `float` | The change in y (Δy).        |

        ## Notes:
        - Uses the formula: `Dy = - (Dx * y^2) / (Dx * y + invariant)`.
        """
        numerator = Dx * y_coordinate ** 2
        denominator = Dx * y_coordinate + invariant
        return - numerator / denominator 
    
    
    def calculate_Dx_from_x_y_Dy(
            self,
            x_coordinate: float, 
            y_coordinate: float,
            Dy: float,
    ) -> float:
        """
        ### Calculates Δx for a given Δy using x and y coordinates.

        ## Parameters:
        | Parameter Name | Type    | Description              |
        |----------------|---------|--------------------------|
        | `x_coordinate` | `float` | The x coordinate.        |
        | `y_coordinate` | `float` | The y coordinate.        |
        | `Dy`           | `float` | The change in y (Δy).    |

        ## Returns:
        | Return Name | Type    | Description                 |
        |-------------|---------|-----------------------------|
        | `Dx`        | `float` | The change in x (Δx).       |

        ## Notes:
        - Uses the formula: `Dx = - (Dy * x) / (y + Dy)`.
        """
        numerator = Dy * x_coordinate
        denominator = y_coordinate + Dy
        return - numerator / denominator

    
    def calculate_Dx_from_y_invariant_Dy(
            self,
            y_coordinate: float, 
            invariant: float, 
            Dy: float,
    ) -> float:
        """
        ### Calculates Δx for a given Δy using y coordinate and invariant.

        ## Parameters:
        | Parameter Name | Type    | Description              |
        |----------------|---------|--------------------------|
        | `y_coordinate` | `float` | The y coordinate.        |
        | `invariant`    | `float` | The invariant (x * y).   |
        | `Dy`           | `float` | The change in y (Δy).    |

        ## Returns:
        | Return Name | Type   | Description                  |
        |-------------|--------|------------------------------|
        | `Dx`        | `float`| The change in x (Δx).        |

        ## Notes:
        - Uses the formula: `Dx = - (invariant * Dy) / (y * (y + Dy))`.
        """
        numerator = invariant * Dy
        denominator = y_coordinate * (y_coordinate + Dy)
        return - numerator / denominator
    
    
    def calculate_Dx_from_x_invariant_Dy(
            self,
            x_coordinate: float, 
            invariant: float, 
            Dy: float,
    ) -> float:
        """
        ### Calculates Δx for a given Δy using x coordinate and invariant.

        ## Parameters:
        | Parameter Name | Type    | Description              |
        |----------------|---------|--------------------------|
        | `x_coordinate` | `float` | The x coordinate.        |
        | `invariant`    | `float` | The invariant (x * y).   |
        | `Dy`           | `float` | The change in y (Δy).    |

        ## Returns:
        | Return Name | Type    | Description                 |
        |-------------|---------|-----------------------------|
        | `Dx`        | `float` | The change in x (Δx).       |

        ## Notes:
        - Uses the formula: `Dx = - (Dy * x^2) / (Dy * x + invariant)`.
        """
        numerator = Dy * x_coordinate ** 2
        denominator = Dy * x_coordinate + invariant
        return - numerator / denominator 
    
    
    def calculate_sigmoid_transition_factors(
            self
        ) -> np.ndarray:
        """
        ### Calculates sigmoid transition factors for smooth trade transitions.

        ## Returns:
        | Return Name                  | Type         | Description                              |
        |------------------------------|--------------|------------------------------------------|
        | `sigmoid_transition_factors` | `np.ndarray` | An array of sigmoid transition factors.  |

        ## Notes:
        - Uses a sigmoid function to generate transition factors for smooth increments between animation frames.
        - Forces the last value to be exactly 1 for consistency.
        """
        x = np.linspace(-6, 6, self.trade_increments)
        sigmoid_transition_factors = 1 / (1 + np.exp(-x))
        sigmoid_transition_factors[-1] = 1.0  # Force the last value to be exactly 1
        return sigmoid_transition_factors


    def handle_output_discrepancies(
            self, 
            output_values: List[float], 
            function_descriptions: List[str], 
            trade_input: Tuple[str, float], 
            state_info: Tuple[float]
    ) -> float:
        """
        ### Handles discrepancies in trade output calculations and logs the process.

        ## Parameters:
        | Parameter Name          | Type                | Description                                                                |
        |-------------------------|---------------------|----------------------------------------------------------------------------|
        | `output_values`         | `List[float]`       | A list of calculated output values for the trade.                          |
        | `function_descriptions` | `List[str]`         | A list of descriptions for the functions that produced the output values.  |
        | `trade_input`           | `Tuple[str, float]` | The input trade type and value (e.g., ('Dx', 100)).                        |
        | `state_info`            | `Tuple[float]`      | The state information (x coordinate, y coordinate, invariant).             |

        ## Returns:
        | Return Name          | Type    | Description                                             |
        |----------------------|---------|---------------------------------------------------------|
        | `output_value`       | `float` | The selected output value after handling discrepancies. |

        ## Notes:
        - Logs input and output values, and function descriptions to a file.
        - Checks for discrepancies in output values and handles them based on predefined rules.
        - Returns the selected output value or raises an error if discrepancies cannot be resolved.
        """
        with open("input_output_logs.txt", "a") as log_file:
            log_file.write(30*"_" + "\n\n")
            input_headers = ["Parameter", "Value"]
            trade_type, trade_value = trade_input
            input_data = [
                ("x_coordinate", f"{state_info[0]:.64f}"),
                ("y_coordinate", f"{state_info[1]:.64f}"),
                ("invariant", f"{state_info[2]:.64f}"),
                (trade_type, f"{trade_value:.64f}")
            ]
            inout_table = tabulate(input_data, headers=input_headers, tablefmt="pretty", colalign=('right', 'left'), disable_numparse=True)
            log_file.write("Input Information:\n" + inout_table + "\n\n")
            formatted_outputs = [f"{delta:.64f}" for delta in output_values]
            table = tabulate(zip(function_descriptions, formatted_outputs), headers=["Function", "Output"], tablefmt="pretty", colalign=('right', 'left'), disable_numparse=True)
            log_file.write("Output Calculations:\n" + table + "\n")
            try:
                if all(delta == output_values[0] for delta in output_values):
                    log_file.write("No discrepancies detected.\n")
                    return output_values[0]
                if any(delta < 0 for delta in output_values) and any(delta > 0 for delta in output_values):
                    raise ValueError("Fatal error: Mixed positive and negative outputs cannot be reconciled.")
                if all(delta < 0 for delta in output_values):
                    least_negative = max(output_values)
                    log_file.write(f"Selected least negative value: {least_negative:.15f}\n")
                    return least_negative
                if all(delta > 0 for delta in output_values):
                    most_positive = max(output_values, key=abs)
                    log_file.write(f"Selected most positive value: {most_positive:.15f}\n")
                    return most_positive
            except ValueError as e:
                log_file.write(f"Processing stopped due to error: {str(e)}\n")
                raise RuntimeError(f"Processing stopped due to error: {str(e)}")
            finally:
                log_file.write("Discrepancy check completed.\n\n")

    
    def calculate_trade(
            self, 
            x_coordinate: float, 
            y_coordinate: float, 
            invariant: float, 
            trade_input: Tuple[str, float],
            increment: int,
    ) -> Dict[str, float]:
        """
        ### Calculates the trade output for a given trade input and increment.

        ## Parameters:
        | Parameter Name      | Type                | Description                                          |
        |---------------------|---------------------|------------------------------------------------------|
        | `x_coordinate`      | `float`             | The x coordinate.                                    |
        | `y_coordinate`      | `float`             | The y coordinate.                                    |
        | `invariant`         | `float`             | The invariant (x * y).                               |
        | `trade_input`       | `Tuple[str, float]` | The input trade type and value (e.g., ('Dx', 100)).  |
        | `increment`         | `int`               | The current increment in the trade process.          |

        ## Returns:
        | Return Name         | Type               | Description                                           |
        |---------------------|--------------------|-------------------------------------------------------|
        | `trade_output`      | `Dict[str, float]` | A dictionary containing the calculated trade output.  |

        ## Notes:
        - Uses sigmoid transition factors to smoothly transition trade values.
        - Calculates output values using appropriate swap functions.
        - Handles discrepancies in output values and returns the selected result.
        """
        swap_input_symbol, swap_input_value = trade_input
        incremental_swap_input_value = self.sigmoid_transition_factors[increment] * swap_input_value
        swap_output_symbol = {'Dx': 'Dy', 'Dy': 'Dx'}[swap_input_symbol]
        swap_output_calculators = {
            'x_and_y': (x_coordinate, y_coordinate, incremental_swap_input_value),
            'x_and_invariant': (x_coordinate, invariant, incremental_swap_input_value),
            'y_and_invariant': (y_coordinate, invariant, incremental_swap_input_value)
        }
        output_values = [self.swap_functions[f'{swap_output_symbol}_from_{swap_input_symbol}'][function](*args) 
                         for function, args in swap_output_calculators.items()]
        state_info = (x_coordinate, y_coordinate, invariant)
        result = self.handle_output_discrepancies(output_values, list(swap_output_calculators.keys()), (swap_input_symbol, incremental_swap_input_value), state_info)
        if isinstance(result, RuntimeError):
            raise result
        return {
            swap_input_symbol: incremental_swap_input_value,
            swap_output_symbol: result,
        }

    
    def make_next_row(
            self
    ) -> None:
        """
        ### Prepares the next row for storing trade results.

        ## Parameters:
        None

        ## Returns:
        None

        ## Notes:
        - Retrieves the last coordinates and marginal rates from the current row.
        - Appends new empty lists to store the next row of results.
        """
        start_row_coordinates = self.coordinates[-1][1][-1]
        start_row_marginal_rate_dy_dx = self.marginal_rate_dy_dx[-1][1][-1]
        start_row_marginal_rate_dx_dy = self.marginal_rate_dx_dy[-1][1][-1]
        self.coordinates.append([start_row_coordinates, []])
        self.marginal_rate_dy_dx.append([start_row_marginal_rate_dy_dx, []])
        self.marginal_rate_dx_dy.append([start_row_marginal_rate_dx_dy, []])
        self.effecive_rate_delta_y_delta_x.append([start_row_marginal_rate_dy_dx])
        self.effecive_rate_delta_x_delta_y.append([start_row_marginal_rate_dx_dy])
        self.swap_amounts.append([])

   
    def process_trades(
            self,
    ) -> None:
        """
        ### Processes all trades specified in the trade actions.

        ## Parameters:
        None

        ## Returns:
        None

        ## Notes:
        - Iterates through each trade action and processes trades for each increment.
        - Calculates new coordinates, marginal rates, effective rates, and swap amounts.
        - Updates the invariant [defensive] and prepares the next row for storing results.
        """
        for i, trade_input in enumerate(self.trade_actions):
            invariant = self.invariant[-1]
            x_coordinate, y_coordinate = self.coordinates[i][0]
            for increment in range(self.trade_increments):
                trade_output = self.calculate_trade(x_coordinate, y_coordinate, invariant, trade_input, increment)
                new_x_coordinate = x_coordinate + trade_output['Dx']
                new_y_coordinate = y_coordinate + trade_output['Dy']
                self.coordinates[i][1].append((new_x_coordinate, new_y_coordinate))
                self.marginal_rate_dy_dx[i][1].append((- new_y_coordinate / new_x_coordinate))
                self.marginal_rate_dx_dy[i][1].append((- new_x_coordinate / new_y_coordinate))
                self.effecive_rate_delta_y_delta_x[i].append(trade_output['Dy'] / trade_output['Dx'])
                self.effecive_rate_delta_x_delta_y[i].append(trade_output['Dx'] / trade_output['Dy'])
                self.swap_amounts[i].append((trade_output['Dx'],trade_output['Dy']))
            self.invariant.append(new_x_coordinate * new_y_coordinate)
            self.make_next_row()

    
    def find_min_max_x(
            self
    ) -> Tuple[float]:
        """
        ### Finds the minimum and maximum x coordinates for plotting.

        ## Parameters:
        None

        ## Returns:
        | Return Name | Type    | Description                                                  |
        |-------------|---------|--------------------------------------------------------------|
        | `min_x`     | `float` | The minimum x coordinate divided by 1.2 for buffer space.    |
        | `max_x`     | `float` | The maximum x coordinate multiplied by 1.2 for buffer space. |

        ## Notes:
        - Iterates through all coordinates to find the min and max x values.
        """
        x_coordinates = [coord[0] for i, j in self.coordinates for coord in [i] + j]
        return min(x_coordinates) / 1.2, max(x_coordinates) * 1.2

    
    def calculate_x_array_from_bounds(
            self
    ) -> np.ndarray:
        """
        ### Calculates an array of x values for plotting based on bounds.

        ## Parameters:
        None

        ## Returns:
        | Return Name | Type         | Description           |
        |-------------|--------------|-----------------------|
        | `x_array`   | `np.ndarray` | An array of x values. |

        ## Notes:
        - Uses geometric spacing between the min and max x values for plotting.
        """
        min_x, max_x = self.find_min_max_x()
        return np.geomspace(min_x, max_x, 1000)

    
    def calculate_y_array_from_x_array(
            self
    ) -> np.ndarray:
        """
        ### Calculates an array of y values corresponding to x values.

        ## Parameters:
        None

        ## Returns:
        | Return Name | Type         | Description           |
        |-------------|--------------|-----------------------|
        | `y_array`   | `np.ndarray` | An array of y values. |

        ## Notes:
        - Uses the initial invariant and x array to calculate corresponding y values.
        """
        return self.invariant[0] / self.x_array

    
    def calculate_derivative_arrays(
            self
    ) -> np.ndarray:
        """
        ### Calculates arrays of derivatives ∂y/∂x and ∂x/∂y.

        ## Parameters:
        None

        ## Returns:
        | Return Name     | Type         | Description               |
        |-----------------|--------------|---------------------------|
        | `dy_dx_array`   | `np.ndarray` | An array of ∂y/∂x values. |
        | `dx_dy_array`   | `np.ndarray` | An array of ∂x/∂y values. |

        ## Notes:
        - Uses the y array and x array to calculate the derivatives.
        """
        dy_dx_array = -self.y_array / self.x_array
        dx_dy_array = np.where(dy_dx_array != 0, 1 / dy_dx_array, 0)
        return dy_dx_array, dx_dy_array


    def draw_arrows(
            self, 
            ax: plt.Axes, 
            start_x: float, 
            start_y: float, 
            end_x: float, 
            end_y: float, 
            color: str,
            arrowstyle: str = '-|>',
            lw: float = 2,
            mutation_scale: float = 20,
            zorder: int = 1
    ) -> None:
        """
        ### Draws arrows between two points on a plot.

        ## Parameters:
        | Parameter Name   | Type       | Description                                  |
        |------------------|------------|----------------------------------------------|
        | `ax`             | `plt.Axes` | The axes on which to draw.                   |
        | `start_x`        | `float`    | The starting x coordinate.                   |
        | `start_y`        | `float`    | The starting y coordinate.                   |
        | `end_x`          | `float`    | The ending x coordinate.                     |
        | `end_y`          | `float`    | The ending y coordinate.                     |
        | `color`          | `str`      | The color of the arrow.                      |
        | `arrowstyle`     | `str`      | The style of the arrow (default is '-|>').   |
        | `lw`             | `float`    | The line width of the arrow (default is 2).  |
        | `mutation_scale` | `float`    | The scale of the arrow (default is 20).      |
        | `zorder`         | `int`      | The z-order of the arrow (default is 1).     |

        ## Returns:
        None

        ## Notes:
        - Uses FancyArrowPatch to draw arrows with specified parameters.
        """
        arrow = FancyArrowPatch((start_x, start_y), (end_x, end_y),
                                arrowstyle=arrowstyle, mutation_scale=mutation_scale,
                                color=color, lw=lw, zorder=zorder)
        ax.add_patch(arrow)

    
    def setup_plot(
            self
    ) -> Tuple[plt.Figure, List[plt.Axes]]:
        """
        ### Sets up the plot layout and axes.

        ## Parameters:
        None

        ## Returns:
        | Return Name | Type             | Description                          |
        |-------------|------------------|--------------------------------------|
        | `fig`       | `plt.Figure`     | The figure object for the plot.      |
        | `axs`       | `List[plt.Axes]` | A list of axes objects for subplots. |

        ## Notes:
        - Configures a figure with four subplots, disabling the axis for the text summary subplot.
        """
        fig = plt.figure(figsize=(20, 12), dpi=300)
        axs = [
            plt.subplot2grid((2, 2), (0, 0), fig=fig),
            plt.subplot2grid((2, 2), (1, 0), fig=fig),
            plt.subplot2grid((2, 2), (1, 1), fig=fig),
            plt.subplot2grid((2, 2), (0, 1), fig=fig)  # Text summary
        ]
        axs[3].axis('off')  # Text summary
        return fig, axs

    
    def plot_implicit_curve(
            self, 
            ax: plt.Axes
    ) -> None:
        """
        ### Plots the implicit curve on a given axes.

        ## Parameters:
        | Parameter Name | Type       | Description                |
        |----------------|------------|----------------------------|
        | `ax`           | `plt.Axes` | The axes on which to plot. |

        ## Returns:
        None

        ## Notes:
        - Plots the x and y arrays on the given axes.
        - Configures axis labels and limits.
        """
        ax.plot(self.x_array, self.y_array, color='#e6edf3ff')
        ax.set_xlabel(rf"${self.token_pair['x']}$", fontsize=14)
        ax.set_ylabel(rf"${self.token_pair['y']}$", fontsize=14)
        ax.set_xlim(min(self.x_array), max(self.x_array))
        ax.set_ylim(min(self.y_array), max(self.y_array))

    
    def plot_trade_points_and_arrows(
            self, 
            ax: plt.Axes, 
            x: float, 
            y: float, 
            next_x: float, 
            next_y: float
    ) -> None:
        """
        ### Plots trade points and arrows between them on a given axes.

        ## Parameters:
        | Parameter Name | Type       | Description                |
        |----------------|------------|----------------------------|
        | `ax`           | `plt.Axes` | The axes on which to plot. |
        | `x`            | `float`    | The initial x coordinate.  |
        | `y`            | `float`    | The initial y coordinate.  |
        | `next_x`       | `float`    | The next x coordinate.     |
        | `next_y`       | `float`    | The next y coordinate.     |

        ## Returns:
        None

        ## Notes:
        - Draws arrows to represent trade movements.
        - Plots points at the initial and next coordinates.
        """
        if x != next_x:
            color = '#ff7b72ff'
            self.draw_arrows(ax, x, y, next_x, y, color=color)
        if y != next_y:
            color = '#79c0ffff'
            self.draw_arrows(ax, next_x, y, next_x, next_y, color=color)
        ax.plot(x, y, 'o', color='white', zorder=4)
        ax.plot(next_x, next_y, 'o', color='white', zorder=4)


    def plot_derivative_curves_and_arrows(
            self, 
            axs: List[plt.Axes], 
            x: float, 
            y: float, 
            next_x: float, 
            next_y: float,
            dx_dy: float, 
            dy_dx: float, 
            next_dx_dy: float, 
            next_dy_dx: float
    ) -> None:
        """
        ### Plots derivative curves and arrows on given axes.

        ## Parameters:
        | Parameter Name | Type             | Description                        |
        |----------------|------------------|------------------------------------|
        | `axs`          | `List[plt.Axes]` | A list of axes on which to plot.   |
        | `x`            | `float`          | The initial x coordinate.          |
        | `y`            | `float`          | The initial y coordinate.          |
        | `next_x`       | `float`          | The next x coordinate.             |
        | `next_y`       | `float`          | The next y coordinate.             |
        | `dx_dy`        | `float`          | The initial ∂x/∂y value.           |
        | `dy_dx`        | `float`          | The initial ∂y/∂x value.           |
        | `next_dx_dy`   | `float`          | The next ∂x/∂y value.              |
        | `next_dy_dx`   | `float`          | The next ∂y/∂x value.              |

        ## Returns:
        None

        ## Notes:
        - Plots derivative curves on specified axes.
        - Draws arrows between specified points on the axes.
        """
        self.plot_derivative_y_x(axs[1], x, y, next_x, next_y)
        self.plot_derivative_x_y(axs[2], x, y, next_x, next_y)
        least_negative_dy_dx = max(dy_dx, next_dy_dx)
        least_negative_dx_dy = max(dx_dy, next_dx_dy)
        self.draw_arrows(axs[1], x, least_negative_dy_dx / 10, next_x, least_negative_dy_dx / 10, color='white', lw=0.75)
        self.draw_arrows(axs[2], y, least_negative_dx_dy / 10, next_y, least_negative_dx_dy / 10, color='white', lw=0.75)

    
    def plot_derivative_y_x(
            self, 
            ax: plt.Axes, 
            x: float, 
            y: float, 
            next_x: float, 
            next_y: float
    ) -> None:
        """
        ### Plots ∂y/∂x curve on a given axes.

        ## Parameters:
        | Parameter Name | Type       | Description                    |
        |----------------|------------|--------------------------------|
        | `ax`           | `plt.Axes` | The axes on which to plot.     |
        | `x`            | `float`    | The initial x coordinate.      |
        | `y`            | `float`    | The initial y coordinate.      |
        | `next_x`       | `float`    | The next x coordinate.         |
        | `next_y`       | `float`    | The next y coordinate.         |

        ## Returns:
        None

        ## Notes:
        - Plots ∂y/∂x values on the given axes.
        - Configures axis labels, limits, and fills between the curve and the x-axis.
        """
        ax.plot(self.x_array, self.dy_dx_array, color='#79c0ffff')
        ax.set_xlabel(rf"${self.token_pair['x']}$", fontsize=14)
        ax.set_ylabel(rf"$\frac{{\partial {self.token_pair['y']}}}{{\partial {self.token_pair['x']}}}$", fontsize=20)
        ax.set_xlim(min(self.x_array), max(self.x_array))
        ax.set_ylim(min(self.dy_dx_array), 0)
        ax.plot(x, -y / x, 'o', color='white', zorder=4)
        ax.plot(next_x, -next_y / next_x, 'o', color='white', zorder=4)
        ax.fill_between(self.x_array, self.dy_dx_array, 0, 
                        where=(self.x_array >= min(x, next_x)) & (self.x_array <= max(x, next_x)),
                        color='#79c0ffff', alpha=0.3)

    
    def plot_derivative_x_y(
        self, 
        ax: plt.Axes, 
        x: float, 
        y: float, 
        next_x: float, 
        next_y: float
    ) -> None:
        """
        ### Plots ∂x/∂y curve on a given axes.

        ## Parameters:
        | Parameter Name | Type       | Description                    |
        |----------------|------------|--------------------------------|
        | `ax`           | `plt.Axes` | The axes on which to plot.     |
        | `x`            | `float`    | The initial x coordinate.      |
        | `y`            | `float`    | The initial y coordinate.      |
        | `next_x`       | `float`    | The next x coordinate.         |
        | `next_y`       | `float`    | The next y coordinate.         |

        ## Returns:
        None

        ## Notes:
        - Plots ∂x/∂y values on the given axes.
        - Configures axis labels, limits, and fills between the curve and the y-axis.
        """
        ax.plot(self.y_array, self.dx_dy_array, color='#ff7b72ff')
        ax.set_xlabel(rf"${self.token_pair['y']}$", fontsize=14)
        ax.set_ylabel(rf"$\frac{{\partial {self.token_pair['x']}}}{{\partial {self.token_pair['y']}}}$", fontsize=20)
        ax.set_xlim(min(self.y_array), max(self.y_array))
        ax.set_ylim(min(self.dx_dy_array), 0)
        ax.plot(y, -x / y, 'o', color='white', zorder=4)
        ax.plot(next_y, -next_x / next_y, 'o', color='white', zorder=4)
        ax.fill_between(self.y_array, self.dx_dy_array, 0, 
                        where=(self.y_array >= min(y, next_y)) & (self.y_array <= max(y, next_y)),
                        color='#ff7b72ff', alpha=0.3)

    
    def configure_axes(
            self, 
            axs: List[plt.Axes]
    ) -> None:
        """
        ### Configures axes appearance for the plot.

        ## Parameters:
        | Parameter Name | Type              | Description                   |
        |----------------|-------------------|-------------------------------|
        | `axs`          | `List[plt.Axes]`  | A list of axes to configure.  |

        ## Returns:
        None

        ## Notes:
        - Sets face color, removes ticks, and tick labels for each axis.
        """
        for ax in axs:
            ax.set_facecolor('#0d1117')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])

    
    def draw_crosshairs(
            self, 
            ax: plt.Axes, 
            x: float, 
            y: float, 
            label_x: float, 
            label_y: float
    ) -> None:
        """
        ### Draws crosshairs at a specified point on the plot.

        ## Parameters:
        | Parameter Name | Type       | Description                         |
        |----------------|------------|-------------------------------------|
        | `ax`           | `plt.Axes` | The axes on which to draw.          |
        | `x`            | `float`    | The x coordinate for the crosshair. |
        | `y`            | `float`    | The y coordinate for the crosshair. |
        | `label_x`      | `float`    | The label for the x coordinate.     |
        | `label_y`      | `float`    | The label for the y coordinate.     |

        ## Returns:
        None

        ## Notes:
        - Draws vertical and horizontal lines at the specified coordinates.
        - Places labels near the crosshair lines based on available space.
        """
        ax.axvline(x, color='gray', linestyle='--', lw=0.5)
        ax.axhline(y, color='gray', linestyle='--', lw=0.5)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        space_left = x - xlim[0]
        space_right = xlim[1] - x
        space_bottom = y - ylim[0]
        space_top = ylim[1] - y
        if space_left > space_right:
            text_x = x - (space_left / 2)
            h_align = 'right'
        else:
            text_x = x + (space_right / 2)
            h_align = 'left'
        if space_bottom > space_top:
            text_y = y - (space_bottom / 2)
            v_align = 'top'
        else:
            text_y = y + (space_top / 2)
            v_align = 'bottom'
        ax.text(x, text_y, f'{label_x}', verticalalignment=v_align, horizontalalignment='center', color='white', fontsize=11, rotation=90, path_effects=[path_effects.withStroke(linewidth=3, foreground='black')])
        ax.text(text_x, y, f'{label_y}', verticalalignment='center', horizontalalignment=h_align, color='white', fontsize=11, path_effects=[path_effects.withStroke(linewidth=3, foreground='black')])

    
    def write_text_summary(
            self, 
            ax: plt.Axes, 
            x: float, 
            y: float, 
            dx_dy: float, 
            dy_dx: float, 
            next_x: float, 
            next_y: float, 
            next_dx_dy: float, 
            next_dy_dx: float, 
            Dx: float, 
            Dy: float, 
            Dx_Dy: float, 
            Dy_Dx: float
    ) -> None:
        """
        ### Writes a text summary of the trade details on a given axes.

        ## Parameters:
        | Parameter Name | Type       | Description                       |
        |----------------|------------|-----------------------------------|
        | `ax`           | `plt.Axes` | The axes on which to write.       |
        | `x`            | `float`    | The initial x coordinate.         |
        | `y`            | `float`    | The initial y coordinate.         |
        | `dx_dy`        | `float`    | The initial ∂x/∂y value.          |
        | `dy_dx`        | `float`    | The initial ∂y/∂x value.          |
        | `next_x`       | `float`    | The next x coordinate.            |
        | `next_y`       | `float`    | The next y coordinate.            |
        | `next_dx_dy`   | `float`    | The next ∂x/∂y value.             |
        | `next_dy_dx`   | `float`    | The next ∂y/∂x value.             |
        | `Dx`           | `float`    | The change in x (Δx).             |
        | `Dy`           | `float`    | The change in y (Δy).             |
        | `Dx_Dy`        | `float`    | The effective Δx/Δy rate.         |
        | `Dy_Dx`        | `float`    | The effective Δy/Δx rate.         |

        ## Returns:
        None

        ## Notes:
        - Writes a formatted table of trade details on the specified axes.
        """
        data = [
            ["",""],
            [f"Initial {self.token_pair['x']} coordinate", f"{x:.8f}"],
            [f"Final {self.token_pair['x']} coordinate", f"{next_x:.8f}"],
            [f"Change in {self.token_pair['x']} (Δ{self.token_pair['x']})", f"{Dx:.8f}"],
            [f"Initial marginal rate ∂{self.token_pair['y']}/∂{self.token_pair['x']}", f"{dy_dx:.8f}"],
            [f"Final marginal rate ∂{self.token_pair['y']}/∂{self.token_pair['x']}", f"{next_dy_dx:.8f}"],
            [f"Effective rate Δ{self.token_pair['y']}/Δ{self.token_pair['x']}", f"{Dy_Dx:.8f}"],
            ["",""],
            [f"Initial {self.token_pair['y']} coordinate", f"{y:.8f}"],
            [f"Final {self.token_pair['y']} coordinate", f"{next_y:.8f}"],
            [f"Change in {self.token_pair['y']} (Δ{self.token_pair['y']})", f"{Dy:.8f}"],
            [f"Initial marginal rate ∂{self.token_pair['x']}/∂{self.token_pair['y']}", f"{dx_dy:.8f}"],
            [f"Final marginal rate ∂{self.token_pair['x']}/∂{self.token_pair['y']}", f"{next_dx_dy:.8f}"],
            [f"Effective rate Δ{self.token_pair['x']}/Δ{self.token_pair['y']}", f"{Dx_Dy:.8f}"]
        ]
        table_str = tabulate(data, headers=['Parameter', 'Value'], tablefmt='plain', colalign=('right', 'right'))
        ax.text(0.5, 0.5, table_str, fontsize=16, ha='center', va='center', transform=ax.transAxes, family='monospace', color='white')


    def get_plot_values(
            self,
            trade_index: int, 
            increment_index: int
    ) -> Tuple[float]:
        """
        ### Retrieves plot values for a specific trade and increment.

        ## Parameters:
        | Parameter Name    | Type   | Description                            |
        |-------------------|--------|----------------------------------------|
        | `trade_index`     | `int`  | The index of the trade action.         |
        | `increment_index` | `int`  | The index of the trade increment.      |

        ## Returns:
        | Return Name       | Type    | Description                           |
        |-------------------|---------|---------------------------------------|
        | `x`               | `float` | The initial x coordinate.             |
        | `y`               | `float` | The initial y coordinate.             |
        | `next_x`          | `float` | The next x coordinate.                |
        | `next_y`          | `float` | The next y coordinate.                |
        | `dx_dy`           | `float` | The initial ∂x/∂y value.              |
        | `dy_dx`           | `float` | The initial ∂y/∂x value.              |
        | `next_dx_dy`      | `float` | The next ∂x/∂y value.                 |
        | `next_dy_dx`      | `float` | The next ∂y/∂x value.                 |
        | `Dx`              | `float` | The change in x (Δx).                 |
        | `Dy`              | `float` | The change in y (Δy).                 |
        | `Dx_Dy`           | `float` | The effective Δx/Δy rate.             |
        | `Dy_Dx`           | `float` | The effective Δy/Δx rate.             |

        ## Notes:
        - Retrieves values from the coordinates, marginal rates, effective rates, and swap amounts.
        """
        x, y = self.coordinates[trade_index][0]
        next_x, next_y = self.coordinates[trade_index][1][increment_index]
        dx_dy = self.marginal_rate_dx_dy[trade_index][0]
        dy_dx = self.marginal_rate_dy_dx[trade_index][0]
        next_dx_dy = self.marginal_rate_dx_dy[trade_index][1][increment_index]
        next_dy_dx = self.marginal_rate_dy_dx[trade_index][1][increment_index]
        Dx, Dy = self.swap_amounts[trade_index][increment_index]
        Dx_Dy = self.effecive_rate_delta_x_delta_y[trade_index][increment_index]
        Dy_Dx = self.effecive_rate_delta_y_delta_x[trade_index][increment_index]
        return x, y, next_x, next_y, dx_dy, dy_dx, next_dx_dy, next_dy_dx, Dx, Dy, Dx_Dy, Dy_Dx


    def plot_implicit_and_price_curves(
            self, 
            axs: List[plt.Axes], 
            trade_index: int, 
            increment_index: int
    ) -> List[plt.Axes]:
        """
        ### Plots implicit and price curves for a specific trade and increment.

        ## Parameters:
        | Parameter Name    | Type             | Description                            |
        |-------------------|------------------|----------------------------------------|
        | `axs`             | `List[plt.Axes]` | A list of axes on which to plot.       |
        | `trade_index`     | `int`            | The index of the trade action.         |
        | `increment_index` | `int`            | The index of the trade increment.      |

        ## Returns:
        | Return Name | Type              | Description                                 |
        |-------------|-------------------|---------------------------------------------|
        | `axs`       | `List[plt.Axes]`  | A list of axes with plotted curves.         |

        ## Notes:
        - Plots the implicit curve, trade points, arrows, and derivative curves.
        - Draws crosshairs and writes a text summary on the specified axes.
        """
        (x, y, next_x, next_y, 
        dx_dy, dy_dx, next_dx_dy, next_dy_dx, 
        Dx, Dy, Dx_Dy, Dy_Dx) = self.get_plot_values(trade_index, increment_index)
        self.plot_implicit_curve(axs[0])
        self.plot_trade_points_and_arrows(axs[0], x, y, next_x, next_y)
        self.plot_derivative_curves_and_arrows(axs, x, y, next_x, next_y, dx_dy, dy_dx, next_dx_dy, next_dy_dx)
        self.draw_crosshairs(axs[0], x, y, f'{x:.4f}', f'{y:.4f}')
        self.draw_crosshairs(axs[0], next_x, next_y, f'{next_x:.4f}', f'{next_y:.4f}')
        self.draw_crosshairs(axs[1], x, dy_dx, f'{x:.4f}', f'{dy_dx:.4f}')
        self.draw_crosshairs(axs[1], next_x, next_dy_dx, f'{next_x:.4f}', f'{next_dy_dx:.4f}')
        self.draw_crosshairs(axs[2], y, dx_dy, f'{y:.4f}', f'{dx_dy:.4f}')
        self.draw_crosshairs(axs[2], next_y, next_dx_dy, f'{next_y:.4f}', f'{next_dx_dy:.4f}')
        self.configure_axes(axs)
        self.write_text_summary(axs[3], x, y, dx_dy, dy_dx, next_x, next_y, next_dx_dy, next_dy_dx, Dx, Dy, Dx_Dy, Dy_Dx)
        return axs

# %% [markdown]
# ## Animated Figure Maker  

# %%
def create_trade_animation(
        token_pair: Dict[str, str] = {'x' : 'FOO', 'y' : 'BAR'},
        x_0: float = 1234,
        y_0: float = 5678,
        trade_actions: List[Tuple[str, float]] = [('Dx', +100), 
                                                  ('Dy', +321), 
                                                  ('Dx', -123), 
                                                  ('Dy', -321),
                                                  ('Dx', +150), 
                                                  ('Dy', +500), 
                                                  ('Dx', -150), 
                                                  ('Dy', -600)],
        trade_increments = 50,
        output_filename='trade_animation.mp4',
) -> None:
    """
    ### Creates an animation of trade actions over time.

    ## Parameters:
    | Parameter Name     | Type                      | Description                                                                      |
    |--------------------|---------------------------|----------------------------------------------------------------------------------|
    | `token_pair`       | `Dict[str, str]`          | A dictionary mapping token identifiers (default is `{'x': 'FOO', 'y': 'BAR'}`).  |
    | `x_0`              | `float`                   | Initial x-coordinate value (default is `1234`).                                  |
    | `y_0`              | `float`                   | Initial y-coordinate value (default is `5678`).                                  |
    | `trade_actions`    | `List[Tuple[str, float]]` | A list of trade actions with direction and value (default is specified actions). |
    | `trade_increments` | `int`                     | Number of increments to divide each trade action into (default is `50`).         |
    | `output_filename`  | `str`                     | Filename to save the resulting animation (default is `'trade_animation.mp4'`).   |

    ## Returns:
    None

    ## Notes:
    - Initializes an instance of `HomeworkOne` with the given parameters.
    - Sets up the plot and prepares the animation using `FuncAnimation`.
    - The `update_frame` function updates the plot for each frame.
    - The animation is saved to the specified `output_filename` using `ffmpeg` writer.
    """
    instance = HomeworkOne(token_pair, x_0, y_0, trade_actions, trade_increments)
    fig, axs = instance.setup_plot()
    total_frames = len(trade_actions) * trade_increments
    def update_frame(frame_index):
        nonlocal axs
        trade_index = frame_index // trade_increments
        increment_index = frame_index % trade_increments
        for ax in axs:
            ax.clear()
            ax.cla()
        axs[3].axis('off')
        axs = instance.plot_implicit_and_price_curves(axs, trade_index, increment_index)
    ani = FuncAnimation(fig, update_frame, frames=total_frames, interval=50, repeat=False)
    ani.save(output_filename, writer='ffmpeg', dpi=300, bitrate=180000)
    plt.close(fig)

# %% [markdown]
# ## Static Figure Maker

# %%
def create_static_trade_plot(
        token_pair: Dict[str, str] = {'x' : 'FOO', 'y' : 'BAR'},
        x_0: float = 1234,
        y_0: float = 5678,
        trade_actions: List[Tuple[str, float]] = [('Dx', +100), 
                                                  ('Dy', +321), 
                                                  ('Dx', -123), 
                                                  ('Dy', -321),
                                                  ('Dx', +150), 
                                                  ('Dy', +500), 
                                                  ('Dx', -150), 
                                                  ('Dy', -600)],
        output_filename: str = 'static_trade_plot.png',
        trade_index: int = 0,
) -> None:
    """
    ### Creates a static plot of a specific trade action.

    ## Parameters:
    | Parameter Name   | Type                      | Description                                                                      |
    |------------------|---------------------------|----------------------------------------------------------------------------------|
    | `token_pair`     | `Dict[str, str]`          | A dictionary mapping token identifiers (default is `{'x': 'FOO', 'y': 'BAR'}`).  |
    | `x_0`            | `float`                   | Initial x-coordinate value (default is `1234`).                                  |
    | `y_0`            | `float`                   | Initial y-coordinate value (default is `5678`).                                  |
    | `trade_actions`  | `List[Tuple[str, float]]` | A list of trade actions with direction and value (default is specified actions). |
    | `output_filename`| `str`                     | Filename to save the resulting plot (default is `'static_trade_plot.png'`).      |
    | `trade_index`    | `int`                     | Index of the trade action to plot (default is `0`).                              |

    ## Returns:
    None

    ## Notes:
    - Initializes an instance of `HomeworkOne` with the given parameters.
    - Sets up the plot and plots the specified trade action and increment.
    - The plot is saved to the specified `output_filename`.
    """
    trade_increments = 1
    increment_index = 0
    instance = HomeworkOne(token_pair, x_0, y_0, trade_actions, trade_increments)
    fig, axs = instance.setup_plot()
    axs = instance.plot_implicit_and_price_curves(axs, trade_index, increment_index)
    fig.savefig(output_filename, dpi=300)
    plt.close(fig)

# %% [markdown]
# ### Example Animation Call
# 
# Known issue: 
# Do not attempt to queue multiple animations. 
# You must restart the program between animations or the renderer will crash.

# %%
create_trade_animation(
        token_pair = {'x' : 'FOO', 'y' : 'BAR'},
        x_0 = 1234,
        y_0 = 5678,
        trade_actions = [('Dx', +100), 
                         ('Dy', +321), 
                         ('Dx', -123), 
                         ('Dy', -321),
                         ('Dx', +150), 
                         ('Dy', +500), 
                         ('Dx', -150), 
                         ('Dy', -600)],
        trade_increments = 80,
        output_filename='trade_animation_01.mp4',
)

# create_trade_animation(
#         token_pair = {'x' : 'FOO', 'y' : 'BAR'},
#         x_0 = 1234,
#         y_0 = 5678,
#         trade_actions = [('Dx', +111.111),  
#                          ('Dx', -222.222),  
#                          ('Dx', +333.333),  
#                          ('Dx', -444.444),  
#                          ('Dx', +555.555),   
#                          ('Dx', -666.666),  
#                          ('Dx', +777.777),  
#                          ('Dx', -888.888)], 
#         trade_increments = 80,
#         output_filename='trade_animation_02.mp4',
# )

# %% [markdown]
# ### Example Figure Call

# %%
create_static_trade_plot(
        token_pair = {'x' : 'FOO', 'y' : 'BAR'},
        x_0 = 1234,
        y_0 = 5678,
        trade_actions = [('Dx', +100), 
                         ('Dy', +321), 
                         ('Dx', -123), 
                         ('Dy', -321),
                         ('Dx', +150), 
                         ('Dy', +500), 
                         ('Dx', -150), 
                         ('Dy', -600)],
        output_filename = 'static_trade_plot.png',
        trade_index = 0,
)

# %%



