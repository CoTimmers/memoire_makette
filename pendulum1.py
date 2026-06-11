import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 20, 1000)
omega = 2.21
zeta = 0.02
omega_d = omega * np.sqrt(1 - zeta**2)

x = np.exp(-zeta * omega * t) * np.sin(omega_d * t)


plt.plot(t, x)
plt.xlabel('Time (s)')
plt.ylabel('Angular displacement (rad)')
plt.title('Damped oscillation of crane payload')
plt.grid(True)
plt.savefig('damped_oscillation.png')
plt.show()


import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 20, 1000)
omega = 2.21
zeta = 0.02
omega_d = omega * np.sqrt(1 - zeta**2)

x = np.exp(-zeta * omega * t) * np.sin(omega_d * t)

plt.figure(figsize=(8, 4))
plt.plot(t, x, color='#C8102E', linewidth=1.5)
plt.axhline(y=0, color='black', linewidth=0.5)
plt.xlabel('Time (s)')
plt.ylabel(r'Angular displacement $\theta$ (rad)')
plt.title('Damped oscillation of crane payload')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('damped_oscillation.png', dpi=300)
plt.show()


import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 20, 1000)
omega = 2.21
A = 1.0

x = A * np.sin(omega * t)

T = 2 * np.pi / omega

fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(t, x, color='black', linewidth=1.5)

ax.axhline(y=A, color='blue', linewidth=0.8, linestyle='--', xmin=0, xmax=1)
ax.axhline(y=-A, color='blue', linewidth=0.8, linestyle='--', xmin=0, xmax=1)
ax.axhline(y=0, color='black', linewidth=0.8)

ax.axvline(x=0, color='black', linewidth=0.8)

ax.annotate('', xy=(T, -0.15), xytext=(0, -0.15),
            arrowprops=dict(arrowstyle='<->', color='black', lw=1.0))
ax.text(T/2, -0.25, '$T$', ha='center', va='top', fontsize=12)

ax.text(-0.3, A, '$A$', ha='right', va='center', fontsize=12, color='blue')
ax.text(-0.3, -A, '$-A$', ha='right', va='center', fontsize=12, color='blue')
ax.text(-0.3, 0.05, '$0$', ha='right', va='center', fontsize=10)

# ax.text(2, A + 0.15, r'$x(t) = A\sin(\omega t)$', fontsize=12)

# ax.text(14, A + 0.15, r'$\omega = \sqrt{\dfrac{k}{m}}$', fontsize=11)

ax.set_xlabel('$t$', fontsize=12)
ax.set_xlim(-0.5, 20)
ax.set_ylim(-1.4, 1.5)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

ax.set_xticks([])
ax.set_yticks([])

ax.annotate('', xy=(20, 0), xytext=(19.5, 0),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.0))
ax.annotate('', xy=(0, 1.4), xytext=(0, 1.3),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.0))

plt.tight_layout()
plt.savefig('pendulum_oscillation.png', dpi=300, bbox_inches='tight', transparent=True)
plt.show()
