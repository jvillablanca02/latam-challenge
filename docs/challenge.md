# Parte 1

## Notebook `.ipnyb` mejoras

### 1. Solucionar problemas de trazado

La función sns.barplot() espera argumentos posicionales. Tuve que agregar los argumentos posicionales x= y y= para corregir los gráficos.

### 2. Limpieza de código

**Added constant variables**

No es necesario declarar el tamaño de la fuente en cada gráfico, se puede hacer una vez al principio.


```python
# Set the default font size
FONT_SIZE = 12
plt.rcParams.update({'font.size': FONT_SIZE})
```

The same can be done with the seaborn theme:

```python
sns.set_theme(style="darkgrid")
```

**Code cleaning**

I also improved the efficiency and documentation of functions, take the `get_period_day` function as an example:

Before

```python
def get_period_day(date):
    date_time = datetime.strptime(date, '%Y-%m-%d %H:%M:%S').time()
    morning_min = datetime.strptime("05:00", '%H:%M').time()
    morning_max = datetime.strptime("11:59", '%H:%M').time()
    afternoon_min = datetime.strptime("12:00", '%H:%M').time()
    afternoon_max = datetime.strptime("18:59", '%H:%M').time()
    evening_min = datetime.strptime("19:00", '%H:%M').time()
    evening_max = datetime.strptime("23:59", '%H:%M').time()
    night_min = datetime.strptime("00:00", '%H:%M').time()
    night_max = datetime.strptime("4:59", '%H:%M').time()
    
    if(date_time > morning_min and date_time < morning_max):
        return 'mañana'
    elif(date_time > afternoon_min and date_time < afternoon_max):
        return 'tarde'
    elif(
        (date_time > evening_min and date_time < evening_max) or
        (date_time > night_min and date_time < night_max)
    ):
        return 'noche'
```

After

```python
def get_period_day(date: str) -> str:
    """
    Determine the period of the day (mañana, tarde, noche) based on the provided datetime string.

    Parameters
    ----------
    date : str
        A string representing the date and time in the format '%Y-%m-%d %H:%M:%S'.

    Returns
    -------
    str
        The period of the day:
        - 'mañana' (morning): 05:00 - 11:59
        - 'tarde' (afternoon): 12:00 - 18:59
        - 'noche' (night): 19:00 - 04:59
        If the date format is invalid, returns 'Invalid date format'.
    """
    try:
        date_time = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
        time_of_day = date_time.time()
        
        if time(5, 0) <= time_of_day < time(12, 0):
            return 'mañana'
        elif time(12, 0) <= time_of_day < time(19, 0):
            return 'tarde'
        else:
            return 'noche'
    except ValueError:
        return 'Invalid date format'
```

**Incorrect Delay Rate Calculation**


The delay rate (%) variable was calculated incorrectly. It was computed as the total number of flights divided by the number of delays, which results in a ratio, not a percentage.

For example, a delay rate of 19 for Houston means that for every 19 total flights, there is 1 delayed flight. This does not accurately represent a percentage measure. Therefore, I corrected the calculation to be the number of delays divided by the total number of flights, which provides a consistent percentage measure for the plots.

**Wrong top 10 features**

The hardcoded top 10 features in the code were not the top 10 features from the feature importance. Also, these features were calculated before the class balancing. It could be better to automatically select the top 10 features from the feature importance, and do this after the class balancing. I choose to keep the original top 10 features because they were enough to successfuly pass the model tests.

### 3. Improve plots

There were plots where the x axis labels were wrongly rotated:

Before

![image](/docs/images/not_rotated_xlabels.png)

After

![image](/docs/images/rotated_xlabels.png)

Also, there were plots where the there are several x values and it would have been better to rotate the plot for better analysis. For example:

Before

![image](/docs/images/not_rotated_plot.png)

After

![image](/docs/images/rotated_plot.png)

Some plots were not rotated nor sorted:

Before

![image](/docs/images/missing_sorting.png)

After

![image](/docs/images/sorted.png)

> [!NOTE] 
> Here the delay rate (%) was calculated correctly as well.

The days of the week were not sorted and were in Spanish in the Delay Rate by Day of the Week plot:

Before

![image](/docs/images/days_not_sorted.png)


After

![image](/docs/images/days_sorted.png)

> [!NOTE] 
> More improvements were made on the notebook, but I decided to not document every improvement. For the full version, check the notebook present in this GitHub repository.

### 4. Model selection

#### Advantages of XGBoost:

1. Popularity and Robustness:

* **Industry Standard**: XGBoost is widely used in industry due to its robust performance and versatility across various types of datasets.
* **Proven Track Record**: It has a proven track record in winning numerous data science competitions and benchmarks.

2. Handling Complex Datasets:

* **Scalability**: XGBoost is designed to handle large-scale datasets efficiently.
* **Advanced Features**: It includes advanced functionalities like handling missing values, regularization, and parallel processing, making it suitable for more complex datasets we might encounter in the future.

#### Consideration for Logistic Regression:

1. Response Time:

* **Faster Predictions**: Logistic Regression models are generally faster in making predictions due to their simplicity.
* **Lower Computational Cost**: They require less computational power, which can be crucial if the server's response time is a critical factor in our application.

2. Training Speed:

* **Quicker Training**: Logistic Regression typically trains faster than XGBoost, especially on smaller datasets. This can be advantageous during the development and tuning phases when rapid iterations are needed.

3. Simplicity:

* **Fewer Hyperparameters**: Logistic Regression has fewer hyperparameters to tune, which can simplify the model development process and reduce the risk of overfitting.

#### Conclusion

While XGBoost offers greater versatility and robustness for future larger and more complex datasets, the choice of Logistic Regression could be justified if the server's response time and computational efficiency are of paramount importance.

**Final Decision:** I chose `XGBoost` with top 10 features and class balancing for its popularity and versatility. However, consider Logistic Regression if server response time becomes a critical factor.

# Part 3: Deployment

For the deployment phase, I used Google Cloud Platform (GCP) services. Specifically, I chose to:

1. Save the Docker container as an artifact in Google Container Registry (GCR), which is GCP's private container image storage.

2. Use Google Cloud Run, a serverless compute platform, to deploy and serve the web application.

This approach offers the following benefits:

- **Scalability**: Cloud Run automatically scales the number of container instances based on incoming traffic, ensuring efficient resource usage.
- **Cost-effectiveness**: You only pay for the actual compute resources used during request processing.
- **Simplicity**: Cloud Run abstracts away much of the underlying infrastructure management, allowing developers to focus on the application code.
- **Fast deployment**: With the container image stored in GCR, deploying updates to Cloud Run is quick and straightforward.

### Model Storage

Instead of saving the model in the GitHub repository, I opted to store it in Google Cloud Storage. This approach is better for the following reasons:

1. **Version Control**: It's easier to manage and update different versions of the model independently from the application code.
2. **Repository Size**: Large model files are kept out of the Git repository, ensuring it stays lean and quicker to clone or pull.
3. **Access Control**: You can set fine-grained permissions on who can access or modify the model.
4. **Runtime Integration**: The application can easily load the model from Cloud Storage during runtime, allowing for model updates without redeploying the entire application.
