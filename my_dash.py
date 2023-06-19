import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import plotly.express as px
import numpy as np
import dash_table
from scipy.stats import shapiro

ruta = "UNdata_Produccion.Neta.Per.Capita.xlsx"
hoja = "UNdata_Export_20181107_20032591"
datos = pd.read_excel(ruta, sheet_name=hoja, header=0)

regiones = ["Northern America", "South America", "Europe", "Asia", "China", "India"]
regiones.append("Todas las regiones")  # Agregar opción "Todas las regiones" al dropdown

muestra = datos[datos["Country or Area"].isin(regiones)].copy()
muestra["Year"] = muestra["Year"].astype(int)
muestra["Region"] = muestra['Country or Area']

describe = pd.DataFrame(columns=["Region", "Mean", "Median", "Std", "IQR", "Min", "Max", "Shapiro-Wilk W", "Shapiro-Wilk p"])

app = dash.Dash(__name__)

# Definir el layout de la aplicación
app.layout = html.Div([
    html.H1("Evidencia de Aprendizaje 4. Taller Análisis Exploratorio de Datos con Python"),
    html.H2("Presentado por: Franklin Gonzalez Sierra"),
    dcc.Dropdown(
        id="region-dropdown",
        options=[{"label": region, "value": region} for region in regiones],
        value=regiones[0]
    ),
    html.H2("Estadísticas Descriptivas"),  # Título de la tabla    
    dash_table.DataTable(
        id="table",
        columns=[{"name": i, "id": i} for i in describe.columns],
        data=[],
        style_cell={'textAlign': 'left'},
    ),

    dcc.Graph(id="line-chart"),
    dcc.Graph(id="prediction-chart"),
    dcc.Graph(id="bar-chart"), 

])

# Definir la respuesta a los cambios en los inputs del usuario
@app.callback(
    Output("line-chart", "figure"), 
    [Input("region-dropdown", "value")]
)
def update_line_chart(region):
    if region == "Todas las regiones":
        datosxregion = muestra.copy()  # Usar todos los datos si se selecciona "Todas las regiones"
    else:
        datosxregion = muestra[muestra["Region"] == region]

    fig = go.Figure()

    # Definir una lista de colores para las líneas del gráfico
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'yellow']

    for i, element in enumerate(datosxregion["Element"].unique()):
        fig.add_trace(go.Scatter(
            x=datosxregion["Year"], 
            y=datosxregion[datosxregion["Element"] == element]["Value"], 
            mode='lines', 
            name=element, 
            line=dict(color=colors[i % len(colors)]),  # Establecer el color de la línea
            hoverinfo='name+y',  # Información que se mostrará al pasar el mouse sobre la línea
        ))

    fig.update_layout(
        title="Correlaciones",  # Agregar título
        hovermode="x unified",  # Hacer que el efecto hover se active en la misma posición x para todas las líneas
        xaxis=dict(rangeslider=dict(visible=True))  # Mostrar un selector de rango en el eje x
    )
    
    return fig


@app.callback(
    Output("prediction-chart", "figure"), 
    [Input("region-dropdown", "value")]
)
def update_prediction_chart(region):
    if region == "Todas las regiones":
        datoxregion = muestra.copy()  # Usar todos los datos si se selecciona "Todas las regiones"
    else:
        datoxregion = muestra[muestra["Region"] == region]

    # Pronóstico para la producción bruta
    gross_data = datoxregion[datoxregion["Element"].str.contains("Gross")]
    X_gross = gross_data["Year"].values.reshape(-1, 1)
    y_gross = gross_data["Value"].values

    gross_model = LinearRegression()
    gross_model.fit(X_gross, y_gross)
    gross_prediction = gross_model.predict(X_gross)

    # Pronóstico para la producción neta
    net_data = datoxregion[datoxregion["Element"].str.contains("Net")]
    X_net = net_data["Year"].values.reshape(-1, 1)
    y_net = net_data["Value"].values

    net_model = LinearRegression()
    net_model.fit(X_net, y_net)
    net_prediction = net_model.predict(X_net)

    # Crear figura
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X_gross.squeeze(), y=y_gross,
                             mode='markers', name='Actual Gross Production'))
    fig.add_trace(go.Scatter(x=X_gross.squeeze(), y=gross_prediction,
                             mode='lines', name='Predicted Gross Production'))
    fig.add_trace(go.Scatter(x=X_net.squeeze(), y=y_net,
                             mode='markers', name='Actual Net Production'))
    fig.add_trace(go.Scatter(x=X_net.squeeze(), y=net_prediction,
                             mode='lines', name='Predicted Net Production'))
    fig.update_layout(title="Predicciones")  # Agregar título

    return fig


@app.callback(
    Output("bar-chart", "figure"), 
    [Input("region-dropdown", "value")]
)
def update_bar_chart(region):
    if region == "Todas las regiones":
        capacidadxindice = muestra.pivot_table(index="Region", columns="Element", values="Value", aggfunc=np.mean)
    else:
        capacidadxindice = muestra[muestra["Region"] == region].pivot_table(index="Region", columns="Element", values="Value", aggfunc=np.mean)
    
    fig = go.Figure([go.Bar(x=capacidadxindice.index, y=capacidadxindice[column], name=column)
                     for column in capacidadxindice.columns])
    fig.update_layout(barmode='group', xaxis_title="Region", yaxis_title="Capacity Index",
                      title="Índices de Capacidad de Exportación, Importación y Producción")
    return fig


@app.callback(
    Output("table", "data"), 
    [Input("region-dropdown", "value")]
)
def update_table(region):
    if region == "Todas las regiones":
        describe_all = pd.DataFrame(columns=["Region", "Mean", "Median", "Std", "IQR", "Min", "Max", "Shapiro-Wilk W", "Shapiro-Wilk p"])

        for region in regiones:
            datos_region = muestra[muestra["Region"] == region]
            for column in datos_region.select_dtypes(include=[np.number]).columns:
                data = datos_region[column].dropna()
                if len(data) >= 3:  # Verificar que el conjunto de datos tenga al menos una longitud de 3
                    mean = data.mean()
                    median = data.median()
                    std = data.std()
                    iqr = data.quantile(0.75) - data.quantile(0.25)
                    min_value = data.min()
                    max_value = data.max()
                    shapiro_stat, shapiro_pvalue = shapiro(data)
                    describe_all = describe_all.append({
                        "Region": region,
                        "Mean": mean,
                        "Median": median,
                        "Std": std,
                        "IQR": iqr,
                        "Min": min_value,
                        "Max": max_value,
                        "Shapiro-Wilk W": shapiro_stat,
                        "Shapiro-Wilk p": shapiro_pvalue
                    }, ignore_index=True)

        return describe_all.to_dict("records")
    else:
        describe = pd.DataFrame(columns=["Region", "Mean", "Median", "Std", "IQR", "Min", "Max", "Shapiro-Wilk W", "Shapiro-Wilk p"])

        datos_region = muestra[muestra["Region"] == region]
        for column in datos_region.select_dtypes(include=[np.number]).columns:
            data = datos_region[column].dropna()
            if len(data) >= 3:  # Verificar que el conjunto de datos tenga al menos una longitud de 3
                mean = data.mean()
                median = data.median()
                std = data.std()
                iqr = data.quantile(0.75) - data.quantile(0.25)
                min_value = data.min()
                max_value = data.max()
                shapiro_stat, shapiro_pvalue = shapiro(data)
                describe = describe.append({
                    "Region": region,
                    "Mean": mean,
                    "Median": median,
                    "Std": std,
                    "IQR": iqr,
                    "Min": min_value,
                    "Max": max_value,
                    "Shapiro-Wilk W": shapiro_stat,
                    "Shapiro-Wilk p": shapiro_pvalue
                }, ignore_index=True)

        return describe.to_dict("records")


# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=True)
