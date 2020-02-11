def tuplify(val, length):
  if type(val) in [tuple, list] and len(val) == length:
    return tuple(val)
  else:
    return tuple(val for _ in range(length))

  
def listify(val, length):
  if type(val) in [tuple, list] and len(val) == length:
    return list(val)
  else:
    return [val for _ in range(length)]


def listwrap(val):
  if type(val) in [tuple, list]:
    return list(val)
  else:
    return [val]
