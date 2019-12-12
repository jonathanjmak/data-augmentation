import shutil
import os

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from glob import glob


def main():
  if not os.path.exists("output"):
    os.makedirs("output")
  train_loss_df = pd.DataFrame(list(range(300)), columns=["epoch"])
  train_acc_df = pd.DataFrame(list(range(300)), columns=["epoch"])
  val_loss_df = pd.DataFrame(list(range(300)), columns=["epoch"])
  val_acc_df = pd.DataFrame(list(range(300)), columns=["epoch"])
  best_val_acc_df = pd.DataFrame(list(range(300)), columns=["epoch"])
  for fn in glob("run_logs/*/log.txt"):
    train_losses, train_accs, val_losses, val_accs, best_val_accs = [], [], [], [], []
    with open(fn) as f:
      lines = [line.strip() for line in f.readlines() if len(line.strip())]
      if len(lines) < 300:
        continue
      for line in lines:
        if line[0] != "[":
          continue
        line = line.replace("/", " ").replace(",", " ").replace("[", " ").replace("]", " ").replace("  ", " ").replace("  ", " ").split(" ")
        epoch, train_loss, train_acc, val_loss, val_acc, best_val_acc = int(line[2]), float(line[4]), float(line[6]), float(line[8]), float(line[10]), float(line[12])
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        best_val_accs.append(best_val_acc)
          
    key = fn.split("/")[1][16:]
    print(fn.split("/")[1], key)
    train_loss_df[key] = train_losses
    train_acc_df[key] = train_accs
    val_loss_df[key] = val_losses
    val_acc_df[key] = val_accs
    best_val_acc_df[key] = best_val_accs

  a4_dims = (11.7, 8.27)
  ax = plt.subplots(figsize=a4_dims)[1]
  g = sns.lineplot(ax=ax, x="epoch", y="loss", hue="variable", data=pd.melt(train_loss_df, ["epoch"], value_name="loss"))
  g.set_title("train loss on data vs time")
  g = g.get_figure()
  g.savefig("output/train_loss.png")
  g.clf()

  ax = plt.subplots(figsize=a4_dims)[1]
  g = sns.lineplot(ax=ax, x="epoch", y="acc", hue="variable", data=pd.melt(train_acc_df, ["epoch"], value_name="acc"))
  g.set_title("train acc on data vs time")
  g = g.get_figure()
  g.savefig("output/train_acc.png")
  g.clf()

  ax = plt.subplots(figsize=a4_dims)[1]
  g = sns.lineplot(ax=ax, x="epoch", y="loss", hue="variable", data=pd.melt(val_loss_df, ["epoch"], value_name="loss"))
  g.set_title("train loss on data vs time")
  g = g.get_figure()
  g.savefig("output/val_loss.png")
  g.clf()

  ax = plt.subplots(figsize=a4_dims)[1]
  g = sns.lineplot(ax=ax, x="epoch", y="acc", hue="variable", data=pd.melt(val_acc_df, ["epoch"], value_name="acc"))
  g.set_title("val acc on cifar vs time")
  g = g.get_figure()
  g.savefig("output/val_acc.png")

  ax = plt.subplots(figsize=a4_dims)[1]
  g = sns.lineplot(ax=ax, x="epoch", y="acc", hue="variable", data=pd.melt(best_val_acc_df, ["epoch"], value_name="acc"))
  g.set_title("val acc on cifar vs time")
  g = g.get_figure()
  g.savefig("output/best_val_acc.png")

  output = ""
  for df, name in zip([train_loss_df, train_acc_df, val_loss_df, val_acc_df, best_val_acc_df], ["train_loss", "train_acc", "val_loss", "val_acc", "best_val_acc"]):
    final_row = df.iloc[-1]
    columns = ["value"] + df.columns.values.tolist()[1:]
    if not len(output):
      output = ",".join(columns) + "\n"
    values = [name] + df.iloc[-1].tolist()[1:]
    output += ",".join(str(val) for val in values) + "\n"

  for key in ["p_cifar", "p_thresholded", "threshold"]:
    vals = []
    for fn in df.columns.values.tolist()[1:]:
      print(fn.rfind(key), fn.find(key))
      if fn.rfind(key) != fn.find(key) or key != "threshold":
        start = fn.rfind(key) + len(key) + 1
        vals.append(fn[start:start+4])
      else:
        vals.append("0.00")
    output += key + "," + ",".join(str(val) for val in vals) + "\n"

  vals = []
  for fn in df.columns.values.tolist()[1:]:
    vals.append(int("augment" in fn))
  output += "augment," + ",".join(str(val) for val in vals) + "\n"
  
  with open("output/final_values.csv", "w") as f:
    f.write(output.strip())


if __name__ == "__main__":
  main()
