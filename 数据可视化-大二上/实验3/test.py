from scipy.stats import kendalltau

# Define the data
col2 = [6.9768, 7.0515, 7.0442, 7.0162, 6.9989, 7.0709, 7.0106, 6.9991, 7.0092, 6.9962]
col3 = [76.2445, 75.9731, 76.0662, 76.3223, 74.0748, 75.7066, 75.9135, 75.6824, 76.2862, 74.9903]
col4 = [1.9714, 1.9781, 1.9826, 1.9689, 2.0530, 1.9901, 1.9836, 1.9826, 1.9725, 2.0148]

# Calculate Kendall's tau for col2 vs col3 and col2 vs col4
kendall_col2_col3, pvalue_col2_col3 = kendalltau(col2, col3)

print("Kendall's tau for col2 vs col3:", kendall_col2_col3, "p-value:", pvalue_col2_col3)


kendall_col2_col4, pvalue_col2_col4 = kendalltau(col2, col4)

print("Kendall's tau for col2 vs col4:", kendall_col2_col4, "p-value:", pvalue_col2_col4)
