# Parte 1

## Notebook `.ipnyb` mejoras

### 1. Solucionar problemas de trazado

La función sns.barplot() espera argumentos posicionales. Tuve que agregar los argumentos posicionales x= y y= para corregir los gráficos.

### 2. Limpieza de código

**Variables constantes agregadas**

No es necesario declarar el tamaño de la fuente en cada gráfico, se puede hacer una vez al principio.


```python
# Set the default font size
FONT_SIZE = 12
plt.rcParams.update({'font.size': FONT_SIZE})
```

Lo mismo se puede hacer con el tema seaborn:

```python
sns.set_theme(style="darkgrid")
```

**Limpieza de código**

También mejoré la eficiencia y la documentación de las funciones, tome la función `get_period_day` como ejemplo:

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

Después

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

**Cálculo incorrecto de la tasa de retraso**


La variable tasa de retraso (%) se calculó incorrectamente. Se calculó como el número total de vuelos dividido por el número de retrasos, lo que da como resultado una proporción, no un porcentaje.

Por ejemplo, una tasa de retraso de 19 para Houston significa que por cada 19 vuelos totales, hay 1 vuelo retrasado. Esto no representa con precisión una medida porcentual. Por lo tanto, corregí el cálculo para que sea el número de retrasos dividido por el número total de vuelos, lo que proporciona una medida porcentual consistente para los gráficos.

**10 funciones principales incorrectas**

Las 10 características principales codificadas en el código no eran las 10 características principales según la importancia de la característica. Además, estas características se calcularon antes del equilibrio de clases. Podría ser mejor seleccionar automáticamente las 10 características principales a partir de la importancia de las características y hacerlo después del equilibrio de clases. Elegí mantener las 10 características principales originales porque fueron suficientes para pasar con éxito las pruebas del modelo.

### 3. Mejorar las parcelas

Hubo gráficos donde las etiquetas del eje x estaban rotadas incorrectamente:

Antes

![image](/docs/images/not_rotated_xlabels.png)

Despues

![image](/docs/images/rotated_xlabels.png)

Además, hubo gráficos en los que hay varios valores de x y hubiera sido mejor rotar el gráfico para un mejor análisis. Por ejemplo:

antes

![image](/docs/images/not_rotated_plot.png)

despues

![image](/docs/images/rotated_plot.png)

Some plots were not rotated nor sorted:

antes

![image](/docs/images/missing_sorting.png)

despues

![image](/docs/images/sorted.png)

> [!NOTA] 
> Aquí la tasa de retraso (%) también se calculó correctamente.

Los días de la semana no estaban ordenados y estaban en español en el gráfico Tasa de retraso por día de la semana:

Antes

![image](/docs/images/days_not_sorted.png)


despues

![image](/docs/images/days_sorted.png)

[!NOTA] Se realizaron más mejoras en el notebook, pero decidí no documentar cada una de ellas. Para la versión completa, consulta el notebook presente en este repositorio de GitHub.

4. Selección del Modelo
Ventajas de XGBoost:
Popularidad y Robustez:
Estándar de la Industria: XGBoost es ampliamente utilizado en la industria debido a su rendimiento robusto y versatilidad en varios tipos de conjuntos de datos.
Historial Comprobado: Tiene un historial comprobado en ganar numerosas competiciones de ciencia de datos y benchmarks.
Manejo de Conjuntos de Datos Complejos:
Escalabilidad: XGBoost está diseñado para manejar conjuntos de datos a gran escala de manera eficiente.
Características Avanzadas: Incluye funcionalidades avanzadas como manejo de valores faltantes, regularización y procesamiento paralelo, lo que lo hace adecuado para conjuntos de datos más complejos que podríamos encontrar en el futuro.
Consideraciones para la Regresión Logística:
Tiempo de Respuesta:
Predicciones Más Rápidas: Los modelos de Regresión Logística generalmente son más rápidos en hacer predicciones debido a su simplicidad.
Menor Costo Computacional: Requieren menos poder computacional, lo cual puede ser crucial si el tiempo de respuesta del servidor es un factor crítico en nuestra aplicación.
Velocidad de Entrenamiento:
Entrenamiento Más Rápido: La Regresión Logística típicamente entrena más rápido que XGBoost, especialmente en conjuntos de datos más pequeños. Esto puede ser ventajoso durante las fases de desarrollo y ajuste cuando se necesitan iteraciones rápidas.
Simplicidad:
Menos Hiperparámetros: La Regresión Logística tiene menos hiperparámetros para ajustar, lo que puede simplificar el proceso de desarrollo del modelo y reducir el riesgo de sobreajuste.
Conclusión
Mientras que XGBoost ofrece mayor versatilidad y robustez para futuros conjuntos de datos más grandes y complejos, la elección de la Regresión Logística podría justificarse si el tiempo de respuesta del servidor y la eficiencia computacional son de suma importancia.

Decisión Final: Elegí XGBoost con las 10 principales características y balanceo de clases por su popularidad y versatilidad. Sin embargo, considera la Regresión Logística si el tiempo de respuesta del servidor se convierte en un factor crítico.

Parte 3: Despliegue
Para la fase de despliegue, utilicé los servicios de Google Cloud Platform (GCP). Específicamente, elegí:

Guardar el contenedor Docker como un artefacto en Google Container Registry (GCR), que es el almacenamiento de imágenes de contenedores privado de GCP.
Usar Google Cloud Run, una plataforma de computación sin servidor, para desplegar y servir la aplicación web.
Este enfoque ofrece los siguientes beneficios:

Escalabilidad: Cloud Run escala automáticamente el número de instancias de contenedores según el tráfico entrante, asegurando un uso eficiente de los recursos.
Rentabilidad: Solo pagas por los recursos de computación utilizados durante el procesamiento de solicitudes.
Simplicidad: Cloud Run abstrae gran parte de la gestión de infraestructura subyacente, permitiendo a los desarrolladores centrarse en el código de la aplicación.
Despliegue Rápido: Con la imagen del contenedor almacenada en GCR, desplegar actualizaciones en Cloud Run es rápido y sencillo.
Almacenamiento del Modelo
En lugar de guardar el modelo en el repositorio de GitHub, opté por almacenarlo en Google Cloud Storage. Este enfoque es mejor por las siguientes razones:

Control de Versiones: Es más fácil gestionar y actualizar diferentes versiones del modelo independientemente del código de la aplicación.
Tamaño del Repositorio: Los archivos grandes del modelo se mantienen fuera del repositorio de Git, asegurando que se mantenga ligero y más rápido de clonar o descargar.
Control de Acceso: Puedes establecer permisos detallados sobre quién puede acceder o modificar el modelo.
Integración en Tiempo de Ejecución: La aplicación puede cargar fácilmente el modelo desde Cloud Storage durante el tiempo de ejecución, permitiendo actualizaciones del modelo sin necesidad de volver a desplegar toda la aplicación.
