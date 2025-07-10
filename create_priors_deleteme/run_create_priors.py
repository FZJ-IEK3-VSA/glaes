#Hier wetermachen und die priors testen

import sys
sys.path.append('/fast/home/l-madeisky/models_IEK_3/IEK3_Models/ethos-installation/ethos_suite_repositories/glaes')

from create_prior import evaluate_AGRICULTURE

evaluate_AGRICULTURE('your_shapefile.shp', 0, 'test_run')