class Ground:
    def __init__(self, sig=None, mur=None, epr=None, gnd_model=None, ionisation_intensity=None, ionisation_model=None):
        """
        sig(float):电导率
        mur(float):相对磁导率
        epr(float):相对介电常数
        gnd_model(str):接地模型("No", "Perfect", "Lossy")
        ionisation_intensity(str):电离强度
        ionisation_model(str):电离模型
        """
        self.sig = sig
        self.mur = mur
        self.epr = epr
        self.gnd_model = gnd_model
        self.ionisation_intensity = ionisation_intensity
        self.ionisation_model = ionisation_model