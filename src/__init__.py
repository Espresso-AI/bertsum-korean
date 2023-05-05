from .dataset import *
from .model import *
from .rouge import *
from .engine import *
from .utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    print("You are on " + str(torch.cuda.get_device_name(device)))
else:
    print("You are on " + str(device).upper())


