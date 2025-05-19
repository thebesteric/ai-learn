import matplotlib.pyplot as plt

# Data
years = ["2018", "2019", "2020", "2021", "2022"]
market_size = [10.1, 14.69, 22.59, 34.87, 62.5]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(years, market_size, marker='o', linestyle='-', color='b')

# Adding titles and labels
plt.title("Global AI Software Market Size (2018-2022)")
plt.xlabel("Year")
plt.ylabel("Market Size (in billion USD)")
plt.grid(True)

# Display the plot
plt.show()
