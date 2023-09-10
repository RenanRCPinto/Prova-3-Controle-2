import control.matlab as ctl
import matplotlib.pyplot as plt
import numpy as np
import math

plt.close('all')

#1-
#gerando a função de transferência do SMA
numG = np.array([0.2])
denG = np.array([0.06,1.2])
G = ctl.tf(numG,denG)
print('SMA = ',G)

#plotando o diagrama de bode com margem de fase e de ganho pro SMA
magG, phaseG, omegaG = ctl.bode(G, margins=True)
plt.show()

#2-
#achando wc e a margem de fase pro SMA
mgG, mfG, wcfG, wcgG = ctl.margin(G)
print('wc (original) = ',wcgG,'rad/s = ',wcgG/6.28,'Hz')
print('margem de fase (original) = ',mfG,'graus')

#3-
#definindo as especificações do projeto
Mp = 0.13
ts = 0.64
zeta = (-np.log(Mp))/np.sqrt(np.pi**2 + (np.log(Mp))**2)
mf = math.atan(2*zeta/np.sqrt(-2*zeta**2 + np.sqrt(4*zeta**4 + 1)))
wc = 4*np.sqrt((1-2*zeta**2)+np.sqrt(4*zeta**4 - 4*zeta**2 +2))/(ts*zeta)
print('\n')
print('Especificações de projeto:')
print('Mp = ',Mp)
print('ts = ',ts,'s')
print ('zeta (desejado) = ',zeta)
print ('margem de fase (desejada) = ',mf * 180/np.pi,'graus')
print ('wc (desejada) = ',wc,'rad/s = ',wc/6.28,'Hz')

thetac = mf - np.pi
print('thetac = ',thetac,'rad')

FTMAwc = ctl.evalfr(G,1j*wc)
modFTMAwc=np.abs(FTMAwc)  
faseFTMAwc=np.angle(FTMAwc)

print('|G(wc)|dB =',20*np.log10(modFTMAwc),'dB')
print('fase G(wc) =',180*faseFTMAwc/np.pi,'graus')

z = wc/np.tan(thetac+(np.pi/2)-faseFTMAwc)
Kp = wc/(modFTMAwc*np.sqrt((z**2)+(wc**2)))
print('z = ',z)
print('Kp = ',Kp)

#definindo o controlador PI
PI = ctl.tf([Kp,Kp*z],[1,0])
print('\nControlador PI obtido = ',PI)

#4-
#plotando o diagrama de bode com margem de fase e de ganho
#pro sistema compensado
Sistcomp = ctl.series(G,PI)
magSistcomp, phaseSistcomp, omegaSistcomp = ctl.bode(Sistcomp, margins=True)
plt.show()
#achando wc e a margem de fase para o sistema compensado
mgSistcomp, mfSistcomp, wcfSistcomp, wcgSistcomp = ctl.margin(Sistcomp)
print('Parâmetros obtidos:')
print('wc (compensada) = ',wcgSistcomp,'rad/s = ',wcgSistcomp/6.28,'Hz')
print('margem de fase (compensada) = ',mfSistcomp,'graus')

#5-
#gerando a FTMF sem controlador
FTMF = ctl.feedback(G,1)
print('\nFTMF sem controlador = ',FTMF)

y_uncomp, t_uncomp = ctl.step(FTMF)

#gerando a FTMF para o sistema compensado
FTMFcomp = ctl.feedback(Sistcomp,1)
ycomp, tcomp = ctl.step(FTMFcomp)
print('FTMF com controlador = ',FTMFcomp)
print('\nSistema em malha fechada compensado (PI): ',ctl.stepinfo(FTMFcomp))

#plotando o gráfico com a comparação
plt.figure()
plt.title('Comparação da resposta ao degrau do sistema sem controlador e com controlador')
plt.plot(t_uncomp,y_uncomp,label='Sistema sem controlador')
plt.plot(tcomp,ycomp,label='Sistema com controlador')
plt.plot([0,0.01,3],[0,1,1],label='Referência')
plt.grid()
plt.ylabel('Variação angular [u.a.]')
plt.xlabel('t [s]')
plt.legend(loc='best')
plt.show()
