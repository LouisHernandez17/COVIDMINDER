import numpy as np
import datetime
import matplotlib.pyplot as plt


class VAR:
    def __init__(self, p, FullData, num_val=0):
        self.p = p
        if num_val == 0:
            self.Data = FullData.values
            self.Data_Validation = None
            self.Observed = FullData.values
        else:
            self.Data = FullData[:-num_val].values
            self.Data_Validation = FullData[-num_val:].values
            self.Observed = FullData[:-num_val].values

        self.B = None
        self.dates = list(FullData.index)[:len(self.Data)]
        self.titles = FullData.columns

    def fit(self):
        k = len(self.Observed[0])
        Y = self.Data[self.p:].transpose()
        Z = np.array([[1] + [self.Data[t - i - 1][j] for i in range(self.p) for j in range(k)] for t in
                      range(self.p, len(self.Data))]).transpose()
        ZZT = np.dot(Z, Z.transpose())
        ZZInv = np.linalg.pinv(ZZT)
        YZtr = np.dot(Y, Z.transpose())
        self.B = np.dot(YZtr, ZZInv)
        Y_pred = np.dot(self.B, Z)
        T = len(self.Observed)
        E = Y - Y_pred

    def predict(self, n, plot=False, savefig=False, path=None):
        k = len(self.Observed[0])
        for m in range(n):
            Z_pred = np.array([[1] + [self.Data[t - i - 1][j] for i in range(self.p) for j in range(k)] for t in
                               range(self.p + 1, len(self.Data) + 1)]).transpose()
            Y_pred = np.dot(self.B, Z_pred)
            self.Data = np.vstack([self.Data, Y_pred[:, -1]])
            self.dates.append(self.dates[-1] + datetime.timedelta(days=1))
        if plot:
            for i in range(len(self.titles)):
                plt.plot(self.dates[:len(self.Observed)], self.Data[:len(self.Observed)][:, i], 'k-',
                         label='Past Observed')
                if (self.Data_Validation == self.Data_Validation).all():
                    plt.plot(self.dates[len(self.Observed):len(self.Observed) + len(self.Data_Validation)],
                             self.Data_Validation[:, i], '-', label='Validation')
                plt.plot(self.dates[len(self.Observed):], self.Data[len(self.Observed):][:, i], '--',
                         label='Prediction')
                plt.legend(framealpha=0)
                plt.xticks(rotation=20)
                plt.title(self.titles[i])
                if savefig:
                    plt.savefig(path + '/' + self.titles[i])
                plt.show()
