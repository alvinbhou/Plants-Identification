import matplotlib.pyplot as plt
import csv, os
import sys

if(len(sys.argv) != 3):
    print("[Usage] python plot.py [log_file_path] [model_name]")
    exit(1)

LOG_FILE = sys.argv[1]
MODEL_NAME = sys.argv[2]

acc = []
val_acc = []
loss = []
val_loss = []
with open(LOG_FILE, 'r') as f:
    re = csv.reader(f, delimiter=',')
    next(re)
    for r in re:
        acc.append(r[1])
        loss.append(r[2])
        val_acc.append(r[3])
        val_loss.append(r[4])

epochs = len(acc)
plt.plot(range(epochs), acc, label="acc")
plt.plot(range(epochs), val_acc, label="val_acc")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.savefig(os.path.join( 'logs', MODEL_NAME+ "_log_acc.png"), bbox_inches='tight', dpi=160)
plt.close()

epochs = len(acc)
plt.plot(range(epochs), loss, label="loss")
plt.plot(range(epochs), val_loss, label="val_loss")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.savefig(os.path.join('logs', MODEL_NAME+ "_log_loss.png"), bbox_inches='tight', dpi=160)
