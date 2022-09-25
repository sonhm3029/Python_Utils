import os
from dotenv import load_dotenv
load_dotenv()


"""
- Define env variable here
- To accessing variables, use: from utils.dotenv-utils import *

"""
# This 2 way using are the same
# name = os.environ['name']
name = os.getenv('name')
