from weather_functions import (
    load_and_clean_data,
    calculate_monthly_averages,
    plot_monthly_temps,
    plot_season_boxplot,
    train_and_evaluate_model
)

# שלבי הריצה
df = load_and_clean_data('weather_data.csv')
monthly_avg_temp, monthly_avg_humidity = calculate_monthly_averages(df)
plot_monthly_temps(monthly_avg_temp)
plot_season_boxplot(df)
train_and_evaluate_model(df)
