from cleanfid import fid
import numpy as np

reals = '/home/manuelladron/projects/lbis_p/attention-based_image_translation/cyclegan/output/cyclegan/s2w_256' \
        '/_test_set_ind_width_256_reals_y'
fakes = '/home/manuelladron/projects/lbis_p/attention-based_image_translation/cyclegan/output/cyclegan/s2w_256' \
        '/_test_set_ind_width_256_x2y'

kids = []
fids = []
for i in range(5):
    score_kid = fid.compute_kid(reals, fakes)
    score_fid = fid.compute_fid(reals, fakes)
    kids.append(score_kid * 100)
    fids.append(score_fid)

print('----------SCORES----------------------')
print(f'Mean FID: {np.array(fids).mean()}, std: {np.array(fids).std()}')
print(f'Mean KID: {np.array(kids).mean()}, std: {np.array(kids).std() * 100}')