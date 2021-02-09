from .load import load_adult, load_credit, load_magic, load_diabetes, load_iris, load_vehicle, load_satellite, load_redwine, load_forestcov
from .load import load_glass, load_haberman, load_mammographic, load_indian_liver, load_segment, load_ecoli, load_optdigits, load_letterrecog

__all__ = ['load_adult',
		   'load_iris',
		   'load_optdigits',
		   'load_satellite',
		   'load_vehicle',
		   'load_segment',
		   'load_redwine',
		   'load_letterrecog',
		   'load_forestcov',
		   'load_ecoli',
           'load_credit',
           'load_magic',
           'load_diabetes',
           'load_glass',
           'load_haberman',
           'load_mammographic',
           'load_indian_liver']
