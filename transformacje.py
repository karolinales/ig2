import math as m
import numpy as np
class Transformacje:
    def __init__(self, model: str = 'wgs 84'):
        #https://en.wikipedia.org/wiki/World_Geodetic_System#WGS84
        if model == "wgs84":
            self.a = 6378137.0 # semimajor_axis
            self.b = 6356752.31424518 # semiminor_axis
        elif model == "grs80":
            self.a = 6378137.0
            self.b = 6356752.31414036
        else:
            raise NotImplementedError(f"{model} model not implemented")
        self.flattening = (self.a - self.b) / self.a
        self.ecc2 = (self.a**2 - self.b**2)/self.a**2
     
    def xyz2plh(self, X,Y,Z):
        """Funkcja służąca do przeliczenia współrzędnych geocentrycznych XYZ na 
        współrzędne geocentryczne krzywoliniwe (Fi, La, h) przy użyciu algorytmu Hirvonena.
        INPUT: 
            X [float] - współrzędna X danego punktu [m]
            Y [float] - współrzędna Y danego punktu [m]
            Z [float] - współrzędna Z danego punktu [m]
            a [float] - dłuższa półos elipsoidy [m]
            e2 [float] - pierwszy mimosród elipsoidy
        OUT:
            phi [float] - szerokosc geodezyjna punktu [rad]
            lam [float] - długosć geodezyjna punktu [rad]
            hel [float] - wyokosć elipsoidalna punktu [m]"""    
        p   = m.sqrt(X**2 + Y**2)          
        phi = m.atan(Z / (p * (1 - self.ecc2)))    
        N = self.a/np.sqrt(1 - (self.ecc2 * np.sin(phi)**2))
        j=0
        phi_1 = phi
        while 1:
            j+=1
            phi_0 = phi_1
            N = self.a/np.sqrt(1 - (self.ecc2 * np.sin(phi)**2))
            h  = (p/m.cos(phi_0))- N
            phi_1 = m.atan(Z/(p *(1 - (self.ecc2 * (N/(N + h))))))
            phi_1 = phi_1
            if abs(phi_1 - phi_0) < (0.0000001/206265):  
                break
        phi = phi_0
        lam   = m.atan(Y/X)
        N = self.a/np.sqrt(1 - (self.ecc2 * np.sin(phi)**2))
        hel = (p/m.cos(phi))- N
        return phi, lam, hel
    
    def sigma(self, phi):
        """Funkcja ta służy do obliczenia długości łuku południka, konieczna do
        przejścia ze współrzędnych krzywoliniowych na płaszczyznę
        odwzorowania Gaussa-Krügera
        INPUT: 
            phi [float] – szerokość geodezyjna danego punktu [rad]
            a [float]– długość dłuższej pół osi elipsoidy [m]
            e2 [float] – pierwszy mimośród elipsoidy 
        OUt:
            sigma [float] – długość łuku południka [m]"""
        A0 = 1 - (self.ecc2/4)-(3*self.ecc2**2/64)-(5*self.ecc2**3/256)
        A2=(3/8)*(self.ecc2 + (self.ecc2**2/4)+((15*self.ecc2**3)/128))
        A4=15/256*(self.ecc2**2+(3*self.ecc2**3/4))
        A6=35*self.ecc2**3/3072
        sigma = self.a * (A0*phi - A2*np.sin(2*phi) + A4*np.sin(4*phi) - A6*np.sin(6*phi))
        return(sigma)
        
    
    def plh2gk(self, phi, lam, lam0 = 21):
        """Przeliczenie współrzędnych krzywoliniowych na wpółrzędne GK
        INPUT:
            phi [float] - szerokosc geodezyjna punktu przeliczanego[rad]\
            lam [float] - długosc geodezyjna punktu przeliczane [rad]
            lam0 [float] - południk odniesienia, wybierany stosowanie do odwzorowania, na które chcemy przejsc później [rad]
            a [float] - dłuższa półos elipsoidy [m]
            e2 [float] - pierwszy mimosród elipsoidy
        OUT:
            XGK [float] - współrzędna X punktu na płaszczyźnie GK [m]
            YGK [float] - współrzędna Y punktu na płaszczyźnie GK [m]"""
        lam0 = lam0 * np.pi/180
        b2 = self.b ** 2
        ep2 = ((self.a ** 2 ) - b2) / b2
        t = np.tan(phi)
        n2 = ep2 * ((np.cos(phi)) ** 2)
        N = self.a/np.sqrt(1 - (self.ecc2 * np.sin(phi)**2))
        si = self.sigma(phi)
        d_lam = lam - lam0
        eta2 = ep2 * np.cos(phi)**2
        XGK = si + ((d_lam**2)/2) * N * np.sin(phi) * np.cos(phi) * (1 + ((d_lam**2)/12) * np.cos(phi)**2 * (5 - t**2 + 9 * eta2 + 4 * eta2**2) + ((d_lam**4)/360) * np.cos(phi)**4 * (61 - 18 * t**2 + t**4 + 270 * eta2 - 330 * eta2 * t**2))
        YGK = d_lam * N * np.cos(phi) * (1 + d_lam**2/6 * np.cos(phi)**2 * (1 - t**2 + eta2) + d_lam**4/120 * np.cos(phi)**4 * (5 - 18 * t**2 + t**4 + 14 * eta2 - 58 * eta2 * t**2))
        return(XGK, YGK)
    
    def gk2pl2(self, xgk, ygk, strefa = 7):
        """ Przeliczenie współrzędnych GK do układu PL-2000
        INPUT:
            XGK [float] - współrzędna X punktu na płaszczyźnie GK [m]
            YGK [float] - współrzędna Y punktu na płaszczyźnie GK [m]
            strefa [int] - strefa w której znajduję sie punkt w układzie PL-2000 [bez jednostek]
        OUT: 
            x2 [float] - współrzędna X puntku w układzie współrzędnych płaskich PL-2000 [m]
            y2 [float] - współrzędna Y puntku w układzie współrzędnych płaskich PL-2000 [m]"""
        x2 = xgk * 0.999923
        y2 = ygk * 0.999923 + strefa * 1000000 + 500000
        return x2, y2
    
    def gk2pl92(self, xgk,ygk):
        """ Przeliczenie współrzędnych GK do układu PL-1992
        INPUT:
            XGK [float] - współrzędna X punktu na płaszczyźnie GK [m]
            YGK [float] - współrzędna Y punktu na płaszczyźnie GK [m]
        OUT: 
            x92 [float] - współrzędna X puntku w układzie współrzędnych płaskich PL-1992 [m]
            y92 [float] - współrzędna Y puntku w układzie współrzędnych płaskich PL-1992 [m]"""
        x92 = xgk * 0.9993 - 5300000
        y92 = ygk * 0.9993 + 500000
        return x92, y92
    
    def plh2xyz(self, phi, lam, hel):
        """Funkcja odwrotna do funckcji xyz2plh. Pozwala ona na przeliczenie współrzędnych geocentyrcznych
        krzywoliniowych na wspoółrzędne geocentryczne prostokątne
        INPUT:
            phi [float] - szerokosc geodezyjna punku [rad]
            lam [float] - długosc geodezyjna punktu [rad]
            hel [float] - wysokosc elipsoidalana punktu
        OUT:
            x [float] - wspołrzędna X punktu [m]
            y [float] - współrzedna Y punktu [m]
            z [float] - wspoółrzędna Z puntku [m]"""
        N = self.a/np.sqrt(1 - (self.ecc2 * np.sin(phi)**2))
        y = (N + hel) * np.cos(phi) * np.sin(lam)
        x = (N + hel) * np.cos(phi) * np.cos(lam)
        z = (N * (1 - self.ecc2)+ hel) * np.sin(phi)
        return x,y,z

    def neu(self,x,y,z):
        """Funkcja służy do obliczenia współrzednych topocentrycznych, przyumujących jako początek ukłądu współrzędnych, punkt 
        o współrzędnych srednich. Funkcja ta oblicza odpowiadnią macierz obrotu układu, a także oblicza współrzędne punktów.
        INPUT: 
            x [array] - tablica zawierająca współrzędne X punktów [m]
            y [array] - tablica zawierająca współrzędne Y punktów [m]
            z [array] - tablica zawierająca współrzędne Z punktów [m]
        OUT:
            delta_neu [array] - tablica o trzech kolumnach oraz zmiennej liczbie wierszy zależnej od ilosci punktów, które 
                                przliczamy. W pierszej kolumnie znajdują się wartosci N, w drugiej E, w trzeciej Z [m]"""
        x_sr = np.average(x)
        y_sr = np.average(y)
        z_sr = np.average(z)
        phi, lam, hel = self.xyz2plh(x_sr, y_sr, z_sr)
        n = [-np.sin(phi) * np.cos(lam), -np.sin(phi) * np.sin(lam), np.cos(phi)]
        e = [-np.sin(lam), np.cos(lam), 0]
        u = [np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)]
        R = np.transpose(np.array([n,e,u]))
        v = np.zeros((len(x),3))
        for i in range(0,len(x)):
            v[i,0] = x[i] - x_sr
            v[i,1] = y[i] - y_sr
            v[i,2] = z[i] - z_sr
        delta_neu = np.zeros(v.shape)
        for a in range(v.shape[0]):
            for b in range(3):
                for c in range(3):
                    delta_neu[a,c] += v[a,b]*R[c,b]
        return delta_neu
    
    def odl2D(self, x1, y1, x2, y2):
        """Funkcja służy do obliczenia odlełosci 2D między dwoma punktami.
        INPUT:
            x1 [float] - współrzędna X punktu pierwszego [m]
            y1 [float] - współrzedna Y punktu pierwszego [m]
            x2 [float] - współrzędna X punktu drugiego [m]
            y2 [float] - współrzedna Y punktu drugiego [m]            
        OUT:
            odl [float] - odlełosc 2D między punktami [m]"""
        odl = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return odl
    
    def odl3D(self, x1, y1, z1, x2, y2, z2):
        """Funkcja służy do obliczenia odlełosci 3D między dwoma punktami.
        INPUT:
            x1 [float] - współrzędna X punktu pierwszego [m]
            y1 [float] - współrzedna Y punktu pierwszego [m]
            z1 [float] - współrzedna Z punktu pierwszego [m]
            x2 [float] - współrzędna X punktu drugiego [m]
            y2 [float] - współrzedna Y punktu drugiego [m]    
            z2 [float] - współrzedna Z punktu drugiego [m]
        OUT:
            odl [float] - odlełosc 3D między punktami [m]"""
        odl = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
        return odl
    
    
    
if __name__ == '__main__':
    # utworzenie obietku 
    geo = Transformacje(model = "grs80")
    # dane XYZ geocentryczne 
    plik = "wsp_inp.txt"
    tablica = np.genfromtxt(plik, delimiter=',', skip_header = 4)
    X = np.array(tablica[:,0])
    Y = np.array(tablica[:,1])
    Z = np.array(tablica[:,2])
    phi = np.zeros_like(X)
    lam = np.zeros_like(X)
    hel = np.zeros_like(X)
    XGK = np.zeros_like(X)
    YGK = np.zeros_like(X)
    x2 = np.zeros_like(X)
    y2 = np.zeros_like(X)
    x92 = np.zeros_like(X)
    y92 = np.zeros_like(X)
    x = np.zeros_like(X)
    y = np.zeros_like(X)
    z = np.zeros_like(X)
    for i in range(0,len(X)):
        phi[i], lam[i], hel[i] = geo.xyz2plh(X[i],Y[i],Z[i])
        XGK[i],YGK[i] = geo.plh2gk(phi[i], lam[i])
        x2[i], y2[i] = geo.gk2pl2(XGK[i], YGK[i])
        x92[i], y92[i] = geo.gk2pl92(XGK[i], YGK[i])
        x[i],y[i],z[i] = geo.plh2xyz(phi[i], lam[i], hel[i])
    neu = geo.neu(X, Y, Z)
    phi, lam = np.rad2deg(phi), np.rad2deg(lam)

    plik = open('wsp_out.txt', 'w')
    plik.write(f'Transformacje współrzędnych \nAutor: Karolina Leszczyńska\nPoniżej zestawienie współrzędnych podanych punktów\n')
    plik.write(f'{"="*60}\n|X\t\t|Y\t\t|Z\t\t|\n')
    for i in range(0,len(X)):
        plik.write(f'|{X[i]:.6f} |{Y[i]:.6f} |{Z[i]:.6f} |\n')
    plik.write(f'{"="*60}\n\tJednostką są stopnie dziesiętne\n|phi\t\t|lam\t\t|hel\t\t|\n')
    for i in range(0,len(X)): 
        plik.write(f'|{phi[i]:.6f}\t|{lam[i]:.6f} \t|{hel[i]:.3f}\t|\n')
    plik.write(f'{"="*60}\n\tUkład PL2000\n|X\t\t|Y\t\t|hel\t\t|\n')
    for i in range(0,len(X)): 
        plik.write(f'|{x2[i]:.3f}\t|{y2[i]:.3f} \t|{hel[i]:.3f}\t|\n')
    plik.write(f'{"="*60}\n\tUkład PL1992\n|X\t\t|Y\t\t|hel\t\t|\n')
    for i in range(0,len(X)): 
        plik.write(f'|{x92[i]:.3f}\t|{y92[i]:.3f} \t|{hel[i]:.3f}\t|\n')    
    plik.write(f'{"="*60}\n\tUkład NEU dla układu o początku w punkcie\nX = {np.average(X):.3f} Y = {np.average(Y):.3f} Z = {np.average(Z):.3f}\n|N\t\t|E\t\t|U\t\t|\n')
    for i in range(0,len(X)): 
        plik.write(f'|{neu[i,0]:.5f}\t|{neu[i,1]:.5f} \t|{neu[i,2]:.5f}\t|\n')
    wybor = input('Czy chcesz obliczyć odległosc miedzy dwoma punktami?\n')
    if wybor == 'tak':
        punkt1 = int(input('Podaj numer pierwszego punku:\n'))
        punkt2 = int(input('Podaj numer drugiego punktu:\n'))
        wybord = input('Chcesz policzyć odległosć 2D czy 3D?\n')
        if wybord == '2D':
            odl = geo.odl2D(x[punkt1], y[punkt1], x[punkt2], y[punkt2])
            plik.write(f'{"="*60}\nOdległosc 2D miedzy punktami {punkt1}, a {punkt2} wynosi: {odl:.3f}m.\n')
            wyb = input('Czy chcesz policzyć jeszcze odległosc 3D?\n')
            if wyb == 'tak':
                odl3D = geo.odl3D(x[punkt1], y[punkt1],z[punkt1], x[punkt2], y[punkt2], z[punkt2])
                plik.write(f'Odleglosc 3D miedzy punktami {punkt1} i {punkt2} wynosi {odl3D:.3f}m.\n')
        if wybord == '3D':
            odl = geo.odl3D(x[punkt1], y[punkt1],z[punkt1], x[punkt2], y[punkt2], z[punkt2])
            plik.write(f'{"="*60}\nOdległosc 3D miedzy punktami {punkt1}, a {punkt2} wynosi: {odl:.3f}m.\n')
            wyb = input('Czy chcesz policzyć jeszcze odległosc 2D?\n')
            if wyb == 'tak':
                odl2D = geo.odl2D(x[punkt1], y[punkt1], x[punkt2], y[punkt2])
                plik.write(f'Odleglosc 2D miedzy punktami {punkt1} i {punkt2} wynosi {odl2D:.3f}m.\n')
    plik.close()