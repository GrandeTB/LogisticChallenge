import numpy as np
import pandas as pd
from pulp import LpMinimize, LpProblem, LpVariable, lpSum, value

# Stock capacity (K1, K2, K3, K4)
capacities = np.array([7500, 3250, 5750, 5000])

# Needs (N1 to N12)
demands = np.array([1250, 800, 925, 1475, 750, 1625, 1100, 975, 1675, 1225, 1050, 1575])

# Travel costs between warehouses S1 to S4 and locations D1 to D12
costs = np.array([
    [53, 62, 48, 53, 85, 93, 103, 87, 123, 115, 85, 90],
    [74, 91, 81, 98, 59, 50, 88, 61, 110, 102, 66, 95],
    [79, 71, 69, 76, 109, 84, 121, 106, 78, 104, 101, 84],
    [80, 106, 69, 87, 123, 70, 67, 73, 89, 76, 96, 78]
])

# Number of warehouses and customers
num_warehouses, num_customers = costs.shape

# Define the problem
prob = LpProblem("Transport_Optimization", LpMinimize)

# Create decision variables: quantities transported from warehouse i to customer j
x = {(i, j): LpVariable(f"x_{i}_{j}", lowBound=0, cat="Integer") for i in range(num_warehouses) for j in range(num_customers)}

# Objective function: minimize total transportation cost
prob += lpSum(costs[i][j] * x[i, j] for i in range(num_warehouses) for j in range(num_customers)), "Total_Transport_Cost"

# Capacity constraints: the total transported from each warehouse cannot exceed its capacity
for i in range(num_warehouses):
    prob += lpSum(x[i, j] for j in range(num_customers)) <= capacities[i], f"Capacity_constraint_Warehouse_S{i+1}"

# Demand constraints: the total received by each customer must satisfy their demand
for j in range(num_customers):
    prob += lpSum(x[i, j] for i in range(num_warehouses)) == demands[j], f"Demand_constraint_Customer_D{j+1}"


prob.solve()

# Optimal or not
if prob.status == 1:  # 1 == 'Optimal'
    # (quantities transported)
    quantities = np.array([[value(x[i, j]) for j in range(num_customers)] for i in range(num_warehouses)])
    
    # Create a DataFrame for better visualization
    quantities_df = pd.DataFrame(quantities, index=[f'S{i+1}' for i in range(num_warehouses)],
                                 columns=[f'D{j+1}' for j in range(num_customers)])
    
    # Display the quantities transported
    print("\nQuantities transported between warehouses and demand locations (units):\n")
    print(quantities_df.to_string())

    
    total_sent = quantities_df.sum(axis=1)
    print("\nTotal quantities sent from each warehouse (units):")
    print(total_sent)

    
    total_received = quantities_df.sum(axis=0)
    print("\nTotal quantities received by each customer (units):")
    print(total_received)

    # Check capacity constraints ! No overload allowed
    print("\nCapacity constraint checks:")
    for i in range(num_warehouses):
        if total_sent.iloc[i] <= capacities[i]:  
            print(f"Warehouse S{i+1}: Capacity constraint satisfied (Sent: {total_sent.iloc[i]}, Capacity: {capacities[i]})")
        else:
            print(f"Warehouse S{i+1}: Capacity constraint NOT satisfied (Sent: {total_sent.iloc[i]}, Capacity: {capacities[i]})")

    # Check the demand constraints 
    print("\nDemand constraint checks:")
    for j in range(num_customers):
        if total_received.iloc[j] == demands[j]: 
            print(f"Customer D{j+1}: Demand constraint satisfied (Received: {total_received.iloc[j]}, Demand: {demands[j]})")
        else:
            print(f"Customer D{j+1}: Demand constraint NOT satisfied (Received: {total_received.iloc[j]}, Demand: {demands[j]})")

    # Final Result : the minimum total transportation cost
    total_cost = value(prob.objective)
    print(f"\nMinimum total cost of transportation: {total_cost}")
else:
    print("Resolution failed.")

#neun­ hund ert ­acht ­und ­zwanzig ­tausend ­fünf ­und ­zwanzig
#