from score_submission import score_subm as ss
from verify_submission import verify_subm as vs
from baseline_algorithm import rec_popular as rp


subm_csv = 'data_Auglaa/submission_popular.csv'
gt_csv = 'data_Auglaa/groundTruth.csv'
test_csv = 'data_Auglaa/test.csv'
data_path = 'data_Auglaa/'

def baseline():
    rp.main(data_path)

def verify():
    vs.main(subm_csv, test_csv)


def score():
    ss.main(gt_csv, subm_csv)

if __name__ == '__main__':
    baseline()
    print('Here')
    verify()
    score()


