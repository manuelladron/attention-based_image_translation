from cleanfid import fid
import numpy as np

reals = '/home/manuelladron/projects/lbis_p/attention-based_image_translation/cyclegan/output/cyclegan/a2o_256' \
        '/_test_set_ind_width_256_reals_x'
fakes = '/home/manuelladron/projects/lbis_p/attention-based_image_translation/cyclegan/output/cyclegan/a2o_256' \
        '/_test_set_ind_width_256_y2x'

kids = []
fids = []
kids_leg = []
fids_leg = []
for i in range(5):
    score_kid_leg = fid.compute_kid(reals, fakes, mode='legacy_pytorch')
    score_fid_leg = fid.compute_fid(reals, fakes, mode='legacy_pytorch')
    score_kid = fid.compute_kid(reals, fakes)
    score_fid = fid.compute_fid(reals, fakes)
    kids.append(score_kid * 100)
    kids_leg.append(score_kid_leg * 100)
    fids.append(score_fid)
    fids_leg.append(score_fid_leg)

print('----------SCORES----------------------')
print(' --- Clean')
print(f'Mean FID: {np.array(fids).mean()}, std: {np.array(fids).std()}')
print(f'Mean KID: {np.array(kids).mean()}, std: {np.array(kids).std() * 100}')
print('\n --- Pytorch Legacy')
print(f'Mean FID: {np.array(fids_leg).mean()}, std: {np.array(fids_leg).std()}')
print(f'Mean KID: {np.array(kids_leg).mean()}, std: {np.array(kids_leg).std() * 100}')