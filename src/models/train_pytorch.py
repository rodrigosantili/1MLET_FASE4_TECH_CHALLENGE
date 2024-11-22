import torch
import torch.optim as optim
import torch.nn as nn
from mlflow_setup import log_metrics, log_pytorch_model


def calculate_accuracy(y_pred, y_true, tolerance=0.05):
    """
    Calcula a acurácia com base em uma tolerância percentual.
    """
    relative_error = torch.abs(y_pred - y_true) / torch.abs(y_true)
    correct = (relative_error < tolerance).float()  # Predições dentro da tolerância
    accuracy = correct.mean().item() * 100  # Percentual de acurácia
    return accuracy


def train_epoch(model, dataloader, criterion, optimizer=None, is_training=False, tolerance=0.05):
    """
    Executa uma época de treinamento ou validação.
    """
    epoch_loss = 0
    epoch_accuracy = 0

    if is_training:
        model.train()
    else:
        model.eval()

    for x_batch, y_batch in dataloader:
        if is_training:
            optimizer.zero_grad()

        # Move os dados para o dispositivo do modelo
        x_batch = x_batch.to(next(model.parameters()).device)
        y_batch = y_batch.to(next(model.parameters()).device)

        # Forward
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)

        # Backward e otimização
        if is_training:
            loss.backward()
            optimizer.step()

        # Cálculo de métricas
        epoch_loss += loss.item()
        epoch_accuracy += calculate_accuracy(y_pred, y_batch, tolerance)

    epoch_loss /= len(dataloader)
    epoch_accuracy /= len(dataloader)

    return epoch_loss, epoch_accuracy


def train_model(model, train_loader, val_loader, epochs=150, lr=0.001, weight_decay=0.01, tolerance=0.05, step_size=40, gamma=0.5, t_max=50):
    """
    Treina o modelo e usa um scheduler para ajustar a taxa de aprendizado.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Adicionar o scheduler
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)

    for epoch in range(epochs):
        # Treinamento
        train_loss, train_accuracy = train_epoch(model, train_loader, criterion, optimizer, is_training=True, tolerance=tolerance)

        # Validação
        val_loss, val_accuracy = train_epoch(model, val_loader, criterion, is_training=False, tolerance=tolerance)

        # Atualizar o scheduler
        scheduler.step()

        # Logar métricas
        log_metrics({
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "learning_rate": scheduler.get_last_lr()[0]
        })

        # Mostrar o progresso a cada 10 épocas
        if (epoch + 1) % 10 == 0 or epoch == 0:  # Também exibe na primeira época
            print(f"Epoch {epoch + 1}/{epochs} | "
                  f"Train Loss: {train_loss:.6f}, Train Accuracy: {train_accuracy:.2f}% | "
                  f"Val Loss: {val_loss:.6f}, Val Accuracy: {val_accuracy:.2f}% | "
                  f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

    log_pytorch_model(model)
    return model
