'''
Temperature Scaling
https://github.com/gpleiss/temperature_scaling
Modified
'''
import torch
from torch import nn, optim
from torch.nn import functional as F
from sklearn.calibration import calibration_curve


class TemperatureScaler(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x):
        return self.temperature_scale(self.model(x))

    def temperature_scale(self, logits):
        temperature = self.temperature.unsqueeze(
            1).expand(logits.size(0), logits.size(1))
        return logits / temperature
    
    def set_temperature(self, valid_loader):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """

        # First: collect all the logits and labels for the validation set
        logits = []
        labels = []
        with torch.no_grad():
            for x, y in valid_loader:
                output = self.model(x)
                logits.append(output)
                labels.append(y)
            logits = torch.cat(logits)
            labels = torch.cat(labels).to(logits.device)
        
        if logits.size(1) == 2:
            nll_criterion = nn.CrossEntropyLoss()
            probas = logits.softmax(1)
        elif logits.size(1) == 1:
            nll_criterion = nn.BCEWithLogitsLoss()
            labels = labels.float().view(-1, 1)
            probas = logits.sigmoid()
        else:
            raise ValueError()

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        fop1, mpv1 = calibration_curve(
            labels.view(-1).cpu().numpy(), probas.cpu().numpy(), n_bins=10)
        print(fop1, mpv1)
        
        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        logits2 = self.temperature_scale(logits)
        if logits.size(1) == 2:
            probas2 = logits2.softmax(1)
        elif logits.size(1) == 1:
            probas2 = logits2.sigmoid()
        after_temperature_nll = nll_criterion(logits2, labels).item()
        fop2, mpv2 = calibration_curve(
            labels.view(-1).cpu().numpy(), 
            probas2.detach().cpu().numpy(), n_bins=10)
        print(fop2, mpv2)
        print('Optimal temperature: %.3f' % self.temperature.item())

        return self


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """

    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super().__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * \
                confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin -
                                 accuracy_in_bin) * prop_in_bin

        return ece
