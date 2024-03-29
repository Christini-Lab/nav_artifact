from exp_data_objects import ExpInformation
from os import listdir, mkdir



def written_cells():
    cell_021022_011_ch3 = ExpInformation(
            cell_file='./data/nav_5-2-2022/220502_011.dat',
            meta_path='./data/nav_5-2-2022/metadata/220502_011',
            channel=3,
            trials={'NaIV0': 1,
                    'NaInact0': 2,

                    'NaIV20': 3,
                    'NaInact20': 4,

                    'NaIV40': 5,
                    'NaInact40': 6,

                    'NaIV60': 7,
                    'NaInact60': 8,

                    'NaIV80': 9,
                    'NaInact80': None})

    #cell_021022_011_ch3.plot_all_iv_traces()
    #cell_021022_011_ch3.plot_all_inact_traces()
    #cell_021022_011_ch3.plot_all_IV()
    #cell_021022_011_ch3.plot_all_inact()
    cell_021022_011_ch3.write_csv('./data/csv/220502_011_ch3')


def save_cell(folder, date, exp_num, channel, trial_nums, ljp, is_saved=False):
    cell = ExpInformation(
        cell_file=f'./data/{folder}/{date}_{exp_num}.dat',
        meta_path=f'./data/{folder}/metadata/{date}_{exp_num}',
        channel=channel,
        ljp=ljp,
        trials={'NaIV0': trial_nums[0],
                'NaInact0': trial_nums[1],

                'NaIV20': trial_nums[2],
                'NaInact20': trial_nums[3],

                'NaIV40': trial_nums[4],
                'NaInact40': trial_nums[5],

                'NaIV60': trial_nums[6],
                'NaInact60': trial_nums[7],

                'NaIV80': trial_nums[8],
                'NaInact80': trial_nums[9]})

    if is_saved:
        cell_folders = listdir('data/csv')
        curr_cell = f'{date}_{exp_num}_ch{channel}'

        if curr_cell not in cell_folders:
            mkdir(f'data/csv/{curr_cell}')
        
        cell.write_csv(f'./data/csv/{date}_{exp_num}_ch{channel}')
        cell.plot_all_iv_traces(f'./data/csv/{date}_{exp_num}_ch{channel}/iv_tr.pdf')
        cell.plot_all_inact_traces(f'./data/csv/{date}_{exp_num}_ch{channel}/in_tr.pdf')
        cell.plot_all_IV(f'./data/csv/{date}_{exp_num}_ch{channel}/iv.pdf')
        cell.plot_all_inact(f'./data/csv/{date}_{exp_num}_ch{channel}/in.pdf')
    else:
        cell.plot_all_iv_traces()
        cell.plot_all_inact_traces()
        cell.plot_all_IV()
        cell.plot_all_inact()


#MEDIUM
#save_cell('nav_5-2-2022', '220502', '008', 2, [3,4,5,6,7,8,9,10,11,12], True)
#save_cell('nav_5-2-2022', '220502', '009', 3, [1,2,3,4,5,6,7,8,None,None], True)
#save_cell('nav_5-2-2022', '220502', '011', 1, [1,2,3,4,5,6,None,None,None,None], True)
#save_cell('nav_5-2-2022', '220502', '011', 3, [1,2,3,4,5,6,7,8,9,None], True)
#save_cell('nav_5-3-2022', '220503', '001', 1, [1,2,3,4,5,6,None,None,None,None], True)
#save_cell('nav_5-3-2022', '220503', '001', 2, [1,2,3,4,5,6,None,None,None,None], True)
#save_cell('nav_5-3-2022', '220503', '002', 1, [1,2,3,4,5,6,None,None,None,None], True)
#save_cell('nav_5-3-2022', '220503', '002', 2, [1,2,3,4,5,6,None,None,None,None], True)
#save_cell('nav_5-3-2022', '220503', '003', 3, [1,2,3,4,5,6,None,None,None,None], True)
#save_cell('nav_5-3-2022', '220503', '004', 1, [1,2,3,4,None,None,None,None,None,None], True)
#save_cell('nav_5-3-2022', '220503', '004', 4, [1,2,3,4,None,None,None,None,None,None], True)
#save_cell('nav_5-3-2022', '220503', '005', 1, [1,2,3,4,5,6,7,8,9,10], True)#One weird trace at 80% compensation!
#save_cell('nav_5-3-2022', '220503', '005', 2, [1,2,3,4,5,6,7,8,9,10], True)
#save_cell('nav_5-3-2022', '220503', '005', 3, [1,2,3,4,5,6,None,None,None,None], True)
#save_cell('nav_5-3-2022', '220503', '005', 4, [1,2,3,4,5,6,7,8,None,None], True)



#HIGH
#save_cell('nav_5-3-2022', '220503', '007', 3, [1,2,3,4,5,6,None,None,None,None], True)
#save_cell('nav_5-3-2022', '220503', '008', 1, [1,2,3,4,5,6,7,8,9,10], True)
#save_cell('nav_5-3-2022', '220503', '008', 3, [1,2,3,4,5,6,7,8,None,None], True)
#save_cell('nav_5-3-2022', '220503', '010', 2, [1,2,3,4,None,None,None,None,None,None], True)
#save_cell('nav_5-3-2022', '220503', '011', 2, [1,2,3,4,5,6,7,8,9,10], True)
#save_cell('nav_5-4-2022', '220504', '001', 1, [1,2,3,4,5,6,None,None,None,None], True)
#save_cell('nav_5-4-2022', '220504', '001', 2, [1,2,3,4,5,6,7,8,9,10], True)
#save_cell('nav_5-4-2022', '220504', '002', 2, [1,2,3,4,5,6,7,8,None,None], True)

#save_cell('nav_5-4-2022', '220504', '006', 3, [1,2,3,4,5,6,None,None,None,None], True)
#save_cell('nav_5-4-2022', '220504', '006', 4, [1,2,3,4,5,6,7,8,None,None], False)




#6/28
#save_cell('nav_6-28-2022', '220628', '001', 1, [1,2,3,4,5,6,7,8,9,10], 15, True)
#save_cell('nav_6-28-2022', '220628', '001', 2, [1,2,3,4,5,6,7,8,9,10], 15, True)
#save_cell('nav_6-28-2022', '220628', '001', 3, [1,2,3,4,5,6,None,None,None,None], 15, True)

#save_cell('nav_6-28-2022', '220628', '002', 1, [1,2,3,4,5,6,7,8,None,None], 15, True)
#save_cell('nav_6-28-2022', '220628', '002', 2, [1,2,3,4,5,6,7,8,None,None], 15, True)

#save_cell('nav_6-28-2022', '220628', '003', 2, [1,2,3,4,5,6,7,8,None,None], 15, True)

#save_cell('nav_6-28-2022', '220628', '004', 1, [1,2,3,4,5,6,7,8,None,None], 15, True)
#save_cell('nav_6-28-2022', '220628', '004', 2, [1,2,3,4,5,6,7,8,9,10], 15, True)

#save_cell('nav_6-28-2022', '220628', '009', 1, [1,2,3,4,5,6,7,8,None,None], 15, True)
#save_cell('nav_6-28-2022', '220628', '009', 2, [1,2,3,4,5,6,7,8,9,10], 15, True)
#save_cell('nav_6-28-2022', '220628', '009', 3, [1,2,3,4,5,6,7,8,9,10], 15, True)

#save_cell('nav_6-28-2022', '220628', '010', 1, [1,2,3,4,5,6,7,8,None,None], 15, True)
#save_cell('nav_6-28-2022', '220628', '010', 2, [1,2,3,4,5,6,None,None,None,None], 15, True)

#save_cell('nav_6-28-2022', '220628', '011', 1, [1,2,3,4,5,6,None,None,None,None], 15, True)
#save_cell('nav_6-28-2022', '220628', '011', 2, [1,2,3,4,5,6,None,None,None,None], 15, True)
#save_cell('nav_6-28-2022', '220628', '011', 4, [1,2,3,4,5,6,7,8,None,None], 15, True)

#save_cell('nav_6-28-2022', '220628', '012', 1, [1,2,3,4,5,6,None,None,None,None], 15, True)
#save_cell('nav_6-28-2022', '220628', '012', 2, [1,2,3,4,5,6,7,8,None,None], 15, True)
#save_cell('nav_6-28-2022', '220628', '012', 4, [1,2,3,4,5,6,7,8,9,10], 15, True)

#save_cell('nav_6-28-2022', '220628', '013', 2, [1,2,3,4,5,6,7,8,None,None], 15, True)

#6/29
#save_cell('nav_6-29-2022', '220629', '001', 2, [1,2,3,4,5,6,7,8,None,None], 15, True)
#save_cell('nav_6-29-2022', '220629', '001', 3, [1,2,3,4,5,6,7,8,9,10], 15, True)
#save_cell('nav_6-29-2022', '220629', '001', 4, [1,2,3,4,5,6,None,None,None,None], 15, True)

#save_cell('nav_6-29-2022', '220629', '002', 2, [1,2,3,4,5,6,7,8,None,None], 15, True)
#save_cell('nav_6-29-2022', '220629', '002', 4, [1,2,3,4,5,6,None,None,None,None], 15, True)

#save_cell('nav_6-29-2022', '220629', '003', 1, [1,2,3,4,5,6,7,8,None,None], 15, True)
#save_cell('nav_6-29-2022', '220629', '003', 3, [1,2,3,4,5,6,None,None,None,None], 15, True)

#save_cell('nav_6-29-2022', '220629', '004', 2, [1,2,3,4,5,6,None,None,None,None], 15, True)
#save_cell('nav_6-29-2022', '220629', '004', 4, [1,2,3,4,5,6,7,8,9,10], 15, True)

#save_cell('nav_6-29-2022', '220629', '005', 2, [1,2,3,4,5,6,7,8,9,10], 15, True)
#save_cell('nav_6-29-2022', '220629', '005', 3, [1,2,3,4,5,6,7,8,None,None], 15, True)

#save_cell('nav_6-29-2022', '220629', '006', 1, [1,2,3,4,5,6,7,8,None,None], 15, True)
#save_cell('nav_6-29-2022', '220629', '006', 4, [1,2,3,4,5,6,7,8,9,10], 15, True)

#save_cell('nav_6-29-2022', '220629', '007', 1, [1,2,3,4,5,6,7,8,None,None], 15, True)
#save_cell('nav_6-29-2022', '220629', '007', 2, [1,2,3,4,5,6,None,None,None,None], 15, True)
#save_cell('nav_6-29-2022', '220629', '007', 3, [1,2,3,4,5,6,None,None,None,None], 15, True)
#save_cell('nav_6-29-2022', '220629', '007', 4, [1,2,3,4,5,6,7,8,None,None], 15, True)

#save_cell('nav_6-29-2022', '220629', '008', 2, [1,2,3,4,5,6,None,None,None,None], 15, True)

#save_cell('nav_6-29-2022', '220629', '009', 1, [1,2,3,4,5,6,7,8,9,10], 15, True)
#save_cell('nav_6-29-2022', '220629', '009', 2, [1,2,3,4,5,6,7,8,None,None], 15, True)
#save_cell('nav_6-29-2022', '220629', '009', 3, [1,2,3,4,5,6,None,None,None,None], 15, True)
#save_cell('nav_6-29-2022', '220629', '009', 4, [1,2,3,4,5,6,7,8,9,10], 15, True)

#save_cell('nav_6-29-2022', '220629', '010', 1, [1,2,3,4,5,6,7,8,9,10], 15, True)
#save_cell('nav_6-29-2022', '220629', '010', 2, [1,2,3,4,5,6,7,8,None,None], 15, True)
#save_cell('nav_6-29-2022', '220629', '010', 3, [1,2,3,4,5,6,7,8,None,None], 15, True)
#save_cell('nav_6-29-2022', '220629', '010', 4, [1,2,3,4,5,6,7,8,9,10], 15, True)

#save_cell('nav_6-29-2022', '220629', '011', 2, [1,2,3,4,5,6,7,8,None,None], 15, True)
#save_cell('nav_6-29-2022', '220629', '011', 3, [1,2,3,4,5,6,None,None,None,None], 15, True)
#save_cell('nav_6-29-2022', '220629', '011', 4, [1,2,3,4,5,6,None,None,None,None], 15, True)

#save_cell('nav_6-29-2022', '220629', '012', 1, [1,2,3,4,5,6,7,8,9,10], 15, True)
#save_cell('nav_6-29-2022', '220629', '012', 2, [1,2,3,4,5,6,7,8,9,10], 15, True)

#save_cell('nav_6-29-2022', '220629', '013', 3, [1,2,3,4,5,6,None,None,None,None], 15, True)
#save_cell('nav_6-29-2022', '220629', '013', 4, [1,2,3,4,5,6,7,8,None,None], 15, True)

#save_cell('nav_6-29-2022', '220629', '014', 1, [1,2,3,4,5,6,7,8,9,10], 15, True)
#save_cell('nav_6-29-2022', '220629', '014', 2, [1,2,3,4,5,6,7,8,9,10], 15, True)
#save_cell('nav_6-29-2022', '220629', '014', 3, [1,2,3,4,5,6,None,None,None,None], 15, True)

#save_cell('nav_6-29-2022', '220629', '014', 1, [1,2,3,4,5,6,7,8,9,10], 15, True)


#6/30
#save_cell('nav_6-30-2022', '220630', '001', 3, [1,2,3,4,5,6,7,8,None,None], 15, True)

save_cell('nav_6-30-2022', '220630', '002', 1, [1,2,3,4,5,6,7,8,None,None], 15, True)
save_cell('nav_6-30-2022', '220630', '002', 2, [1,2,3,4,5,6,7,8,9,10], 15, True)
#save_cell('nav_6-30-2022', '220630', '002', 3, [1,2,3,4,5,6,7,8,9,10], 15, True)
#save_cell('nav_6-30-2022', '220630', '002', 4, [1,2,3,4,5,6,7,8,9,10], 15, True)

#save_cell('nav_6-30-2022', '220630', '004', 1, [1,2,3,4,5,6,7,8,9,10], 15, True)
#save_cell('nav_6-30-2022', '220630', '004', 2, [1,2,3,4,5,6,None,None,None,None], 15, True)
#save_cell('nav_6-30-2022', '220630', '004', 4, [1,2,3,4,5,6,7,8,None,None], 15, True)

#save_cell('nav_6-30-2022', '220630', '005', 2, [1,2,3,4,5,6,7,8,9,10], 15, True)

#save_cell('nav_6-30-2022', '220630', '006', 2, [1,2,3,4,5,6,None,None,None,None], 15, True)

#save_cell('nav_6-30-2022', '220630', '007', 1, [1,2,3,4,5,6,None,None,None,None], 15, True)
#save_cell('nav_6-30-2022', '220630', '007', 3, [1,2,3,4,5,6,7,8,9,10], 15, True)
#save_cell('nav_6-30-2022', '220630', '007', 4, [1,2,3,4,5,6,7,8,9,10], 15, True)

#save_cell('nav_6-30-2022', '220630', '008', 1, [1,2,3,4,5,6,7,8,None,None], 15, True)
#save_cell('nav_6-30-2022', '220630', '008', 2, [1,2,3,4,5,6,7,8,None,None], 15, True)
#save_cell('nav_6-30-2022', '220630', '008', 3, [1,2,3,4,5,6,None,None,None,None], 15, True)
#save_cell('nav_6-30-2022', '220630', '008', 4, [1,2,3,4,5,6,7,8,None,None], 15, True)
#
#save_cell('nav_6-30-2022', '220630', '009', 1, [1,2,3,4,5,6,7,8,9,10], 15, True)


