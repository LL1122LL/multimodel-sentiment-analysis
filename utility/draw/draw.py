import matplotlib.pyplot as plt

def plot_loss_accuracy(train_loss, train_accuracy, val_loss, val_accuracy):
    
    
    # 创建两个子图
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

    # 子图1：训练和验证损失
    axes[0].plot(train_loss, label='Train Loss')
    axes[0].plot(val_loss, label='Validation Loss')
    axes[0].set_xlabel('Batch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    # 子图2：训练和验证准确度
    axes[1].plot(train_accuracy, label='Train Accuracy', marker='o')
    axes[1].plot(val_accuracy, label='Validation Accuracy', marker='o')
    axes[1].set_xlabel('epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()

    # 调整布局
    plt.tight_layout()

    # 显示图形
    plt.show()
