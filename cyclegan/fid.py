from cleanfid import fid


reals = '/home/manuelladron/projects/lbis_p/attention-based_image_translation/cyclegan/output/cyclegan' \
        '/a2o_perc' \
        '/_test_set_ind_reals'
fakes = '/home/manuelladron/projects/lbis_p/attention-based_image_translation/cyclegan/output/cyclegan' \
        '/a2o_perc' \
        '/_test_set_ind'
score = fid.compute_kid(reals, fakes)

print('scores: ', score)