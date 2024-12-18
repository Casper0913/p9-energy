
# Plot the entire test set
plt.figure(figsize=(14, 7))
plt.plot(true_values, label='Actual', color='blue', alpha=0.6)
plt.plot(predictions, label='Predicted',
         linestyle='--', color='red', alpha=0.6)
plt.title("Predicted vs Actual Energy Consumption")
plt.xlabel("Hour")
plt.ylabel("Energy Consumption (kWh)")
plt.legend()
plt.grid(True)
plt.savefig('Training_data/energy_transformer_predictions_full.png', dpi=900)


# Scatter plot of the actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(true_values, predictions, alpha=0.5)
plt.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)],
         color='red', linewidth=2, label="Ideal (y = x)")
plt.xlabel("Actual Consumption (kWh)")
plt.ylabel("Predicted Consumption (kWh)")
plt.title("Prediction vs Actual")
plt.legend()
plt.grid(True)
plt.savefig('Training_data/energy_transformer_scatter.png', dpi=900)


# Save the predictions and true values to a CSV file
results = pd.DataFrame({'Actual': true_values.flatten(),
                       'Predicted': predictions.flatten()})
results.to_csv('Training_data/energy_transformer_predictions.csv', index=False)
