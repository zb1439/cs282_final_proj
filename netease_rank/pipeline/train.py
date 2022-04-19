from datetime import datetime
import logging
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import MultiStepLR

from netease_rank.model import LOSS


class Trainer:
    def __init__(self, cfg, data_source, model, evaluators={}):
        self.lr = cfg.TRAINING.LR
        self.epoch = cfg.TRAINING.EPOCHS
        self.bs = cfg.TRAINING.BATCH_SIZE
        self.cur_epoch = 0
        self.data_source = data_source
        self.model = model
        self.evaluators = evaluators
        self.eval_epoch = cfg.TRAINING.EVAL_EPOCH
        self.info_iter = cfg.TRAINING.INFO_ITER
        tensorboard_dir = os.path.join(os.getcwd(), "tensorboard")
        os.makedirs(tensorboard_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=tensorboard_dir)

        if cfg.TRAINING.OPTIM.upper() == "SGD":
            self.optimizer = SGD(model.parameters(), self.lr)
        elif cfg.TRAINING.OPTIM.upper() == "ADAM":
            self.optimizer = Adam(model.parameters(), self.lr)
        elif cfg.TRAINING.OPTIM.upper() == "ADAMW":
            self.optimizer = AdamW(model.parameters(), self.lr)
        else:
            raise NotImplementedError(f"{cfg.TRAINING.OPTIM} not implemented")

        lr_type = cfg.TRAINING.LR_SCHEDULER.NAME
        if lr_type == "Dummy":
            self.lr_scheduler = MultiStepLR(self.optimizer, milestones=[self.epoch + 1,])
        elif lr_type == "MultiStepLR":
            self.lr_scheduler = MultiStepLR(self.optimizer, milestones=cfg.TRAINING.LR_SCHEDULER.STEPS)
        else:
            raise NotImplementedError(f"{lr_type} not implemented")

        self.criteria = LOSS.get(cfg.TRAINING.CRITERIA.NAME)(cfg)

        if cfg.TRAINING.RESUME:
            self.resume()

        if torch.cuda.is_available():
            self.model.cuda()
            self.criteria.cuda()

    def resume(self):
        logging.info("Loading from local checkpoint...")
        epochs = sorted(list(map(
            lambda x: int(x[11:-4]),
            filter(lambda x: x.startswith("checkpoint"), os.listdir(os.getcwd()))
        )))
        if len(epochs) == 0:
            logging.warning("No checkpoint files found under the current directory: ", os.getcwd())
        state_dict = torch.load(os.path.join(os.getcwd(), f"checkpoint_{epochs[-1]}.pth"))
        self.cur_epoch = state_dict["epoch"]
        self.lr_scheduler.load_state_dict(state_dict["lr_scheduler"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.model.load_state_dict(state_dict["model"])

    def save(self, metrics=None):
        state_dict = {"epoch": self.cur_epoch, "lr_scheduler": self.lr_scheduler.state_dict(),
                      "optimizer": self.optimizer.state_dict()}
        if metrics is not None:
            state_dict.update(metrics)
        if self.model.device != "cpu":
            state_dict["model"] = self.model.cpu().state_dict()
            self.model = self.model.cuda()
        torch.save(state_dict, os.path.join(os.getcwd(), "checkpoint.pth"))

    def train(self):
        loader = DataLoader(self.data_source, batch_size=self.bs, shuffle=True, num_workers=4)
        start_epoch = self.cur_epoch
        for epoch in range(start_epoch, self.epoch):
            self.cur_epoch = epoch
            for i, data in enumerate(loader):
                self.optimizer.zero_grad()

                user_feat, item_feat, scores = data
                if torch.cuda.is_available():
                    user_feat = user_feat.cuda()
                    item_feat = item_feat.cuda()
                    scores = scores.cuda()
                pred_scores = self.model(user_feat, item_feat)
                loss = self.criteria(pred_scores, scores)
                loss.backward()
                self.optimizer.step()

                self.writer.add_scalar("training loss", loss.item(), epoch * len(loader) + i)
                if (i + 1) % self.info_iter == 0:
                    msg = "{} [EPOCH {} ITER {}] Loss: {:2.4f} ".format(
                        datetime.now().strftime("%H:%M:%S"), epoch, i, loss.item())
                    top3_occur = ((torch.argsort(pred_scores[:, :3], dim=1, descending=True) < 3).sum()
                                  / len(pred_scores)).mean()
                    self.writer.add_scalar("top3 co-occurence", top3_occur.item(), epoch * len(loader) + i)
                    msg += "Top3 Co-occurence: {:1.2f}".format(top3_occur.item())
                    logging.info(msg)

            self.lr_scheduler.step(epoch)
            if (epoch + 1) % self.eval_epoch == 0:
                metrics = {}
                for name, evaluator in self.evaluators.items():
                    metric = evaluator.eval(self.model, self.data_source)
                    metrics[name] = metric
                    logging.info("[EPOCH {}] {}: {:2.4f}".format(epoch, name, metric))
                    self.writer.add_scalar(name, metric, (epoch + 1) * len(loader) - 1)
                self.save(metrics)
