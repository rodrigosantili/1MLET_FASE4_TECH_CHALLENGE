import torch


def save_model_local(model, path='model.pth'):
    """
    Salva o modelo treinado localmente no formato PyTorch.
    """
    torch.save(model.state_dict(), path)
    print(f"Modelo salvo em {path}")


# def save_model_to_bucket(model, bucket, path='model.pth', aws_access_key=None, aws_secret_key=None):
#     """
#     Salva o modelo treinado em um bucket de armazenamento.
#
#     Parâmetros:
#         model: o modelo PyTorch a ser salvo.
#         bucket (str): nome do bucket.
#         path (str): caminho e nome do arquivo no bucket.
#         aws_access_key (str): chave de acesso AWS (opcional).
#         aws_secret_key (str): chave secreta AWS (opcional).
#     """
#     import boto3
#     import io
#
#     # Salvar o modelo em um buffer de memória
#     buffer = io.BytesIO()
#     torch.save(model.state_dict(), buffer)
#     buffer.seek(0)
#
#     # Configurar o cliente S3
#     s3 = boto3.client(
#         's3',
#         aws_access_key_id=aws_access_key,
#         aws_secret_access_key=aws_secret_key
#     )
#
#     # Fazer o upload do modelo para o bucket
#     s3.put_object(Bucket=bucket, Key=path, Body=buffer.getvalue())
#     print(f"Modelo salvo em s3://{bucket}/{path}")
