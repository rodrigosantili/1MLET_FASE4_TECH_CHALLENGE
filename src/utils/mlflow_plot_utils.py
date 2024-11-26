import mlflow
import matplotlib.pyplot as plt

def log_plot_to_mlflow(plot_func, filename="plot.png", **kwargs):
    """
    Salva um gráfico gerado por uma função no MLflow.

    Parameters:
        plot_func (callable): Função que gera o gráfico usando matplotlib.
        filename (str): Nome do arquivo para salvar o gráfico.
        **kwargs: Parâmetros adicionais para a função do gráfico.
    """
    # Gera o gráfico
    plot_func(**kwargs)

    # Salva o gráfico localmente
    filepath = f"./{filename}"
    plt.savefig(filepath)
    plt.close()

    # Registra o gráfico como artefato no MLflow
    mlflow.log_artifact(filepath)
    print(f"Gráfico salvo como artefato: {filepath}")