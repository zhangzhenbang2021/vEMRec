aws s3 cp s3://janelia-cosem-datasets/jrc_mus-kidney-2/jrc_mus-kidney-2.n5/em/fibsem-uint8/s3/ ../../raw_data/jrc_mus-kidney-2/em/fibsem-uint8/s3/ --recursive --no-sign-request
aws s3 cp s3://janelia-cosem-datasets/jrc_mus-sc-zp104a/jrc_mus-sc-zp104a.n5/em/fibsem-uint8/s3/ ../../raw_data/jrc_mus-sc-zp104a/em/fibsem-uint8/s3/ --recursive --no-sign-request
aws s3 cp s3://janelia-cosem-datasets/jrc_mus-liver-2/jrc_mus-liver-2.n5/em/fibsem-uint8/s3/ ../../raw_data/jrc_mus-liver-2/em/fibsem-uint8/s3/ --recursive --no-sign-request
aws s3 cp s3://janelia-cosem-datasets/jrc_mus-sc-zp105a/jrc_mus-sc-zp105a.n5/em/fibsem-uint8/s2/ ../../raw_data/jrc_mus-sc-zp105a/em/fibsem-uint8/s2/ --recursive --no-sign-request


python openog.py