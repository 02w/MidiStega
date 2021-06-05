import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from sklearn.metrics import classification_report
from sklearn.preprocessing import scale

from dataset import DataModule, RawDataset
from runner import ResNetRunner


def train():
    print('Training...')
    dm = DataModule('midi', batch_size=5)
    model = ResNetRunner(
        in_channels=4,
        n_feature_maps=64,
        n_classes=2,
        kernel_size=[8, 5, 3]
    )

    checkpoint = ModelCheckpoint(
        dirpath='versions/midi/checkpoints',
        monitor='Validation F1',
        mode='max',
        save_top_k=20,
        save_last=True
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(
        max_epochs=150,
        callbacks=[checkpoint, lr_monitor],
        stochastic_weight_avg=True,
        gpus=1,
        weights_summary='full'
    )
    trainer.fit(model, dm)

    # save best_k_models to a yaml file
    checkpoint.to_yaml()

    return model, checkpoint.best_model_path


def test(model=None, use_ckpt=False):
    print('Testing...')
    data = RawDataset.load('midi.joblib')
    if use_ckpt:
        print(model)
        model = ResNetRunner.load_from_checkpoint(model)
    # pred = model.predict([np.delete(i[0], 0, axis=1).T for i in data.test_data])
    pred = model.predict([scale(i[0]).T for i in data.test_data])
    print(classification_report([i[1] for i in data.test_data], [data.labels[i] for i in pred], digits=3))


def export(model=None, use_ckpt=False, save='model.pt'):
    if use_ckpt:
        print(model)
        model = ResNetRunner.load_from_checkpoint(model)
    
    script = model.to_torchscript(save, method='trace')
    return script


if __name__ == '__main__':
    pl.seed_everything(17)

    model, best_path = train()
    test(model)
    test(model=best_path, use_ckpt=True)

    export(model='versions/midi/checkpoints/epoch=94-step=2469.ckpt', use_ckpt=True, save='stega.pt')
    # model = torch.jit.load('model.pt')
