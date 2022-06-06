def persist_list(list, file_name):
  textfile = open(file_name, "w")
  for element in list:
    textfile.write(str(element) + "\n")
  textfile.close()

def persist_tuple_prefix(arg_tuple, prefix):
  for e in arg_tuple:
    if type(e) is tuple:
      flag = True
      for sub in e:
        if flag:
          persist_list(sub, prefix + "accuracy_train.txt")
          flag = False
        else:
          persist_list(sub, prefix + "accuracy_valid.txt")
    else:
      persist_list(e, prefix + "loss.txt")