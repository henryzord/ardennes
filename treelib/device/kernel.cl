#define LOCAL_WORK_GROUP_SIZE 32

#define TRUE 1
#define FALSE 0

#define LEFT 0
#define RIGHT 1
#define TERM 2
#define ATTR 3
#define THRES 4

#define bool char

//int next_node(int current_node, int go_left, int finished) {
//    return (!finished * ((current_node * 2) + 1 + go_left)) + (finished * current_node);
//}
//

float at(__global float *table, int n_attributes, int x, int y) {
    return table[(n_attributes * x) + y];
}


float entropy_by_index(
    __global float *dataset, int n_objects, int n_attributes,
    __global int *subset_index, int n_classes) {

    int k, i;
    const int class_index = n_attributes - 1;
    float count_class, entropy = 0, subset_size = 0;


    for(i = 0; i < n_objects; i++) {
        subset_size += (float)subset_index[i];
    }

    for(k = 0; k < n_classes; k++) {
        count_class = 0;
        for(i = 0; i < n_objects; i++) {
            count_class += subset_index[i] * (k == at(dataset, n_attributes, i, class_index));
        }
        entropy += select((count_class / subset_size) * log2(count_class / subset_size), (float)0, count_class <= 0);
    }

    return -entropy;
}

float device_gain_ratio(
    __global float *dataset, int n_objects, int n_attributes,
    __global int *subset_index, int attribute_index,
    float candidate,
    int n_classes, float subset_entropy) {

    const int idx = get_global_id(0);

    int j, k;
    const int class_index = n_attributes - 1;

    float   left_entropy = 0, right_entropy = 0,
            left_branch_size = 0, right_branch_size = 0,
            left_count_class, right_count_class,
            subset_size = 0, sum_term,
            is_left, is_from_class;

    for(k = 0; k < n_classes; k++) {
        left_count_class = 0; right_count_class = 0;
        left_branch_size = 0; right_branch_size = 0;
        subset_size = 0, is_from_class = 0, is_left = 0;

        for(j = 0; j < n_objects; j++) {
            is_left = (float)dataset[j * n_attributes + attribute_index] <= candidate;
            is_from_class = (float)(fabs(dataset[j * n_attributes + class_index] - k) < 0.01);

            left_branch_size += is_left * subset_index[j];
            right_branch_size += !is_left * subset_index[j];

            left_count_class += is_left * subset_index[j] * is_from_class;
            right_count_class += !is_left * subset_index[j] * is_from_class;

            subset_size += (float)(subset_index[j]);
        }
        left_entropy += select((left_count_class / left_branch_size) * log2(left_count_class / left_branch_size), (float)0, left_count_class <= 0);
        right_entropy += select((right_count_class / right_branch_size) * log2(right_count_class / right_branch_size), (float)0, right_count_class <= 0);
    }

    sum_term =
        ((left_branch_size / subset_size) * -left_entropy) +
        ((right_branch_size / subset_size) * -right_entropy);

    float
        info_gain = subset_entropy - sum_term,
        split_info = -(
            select((left_branch_size / subset_size)*log2(left_branch_size / subset_size), (float)0, left_branch_size <= 0) +
            select((right_branch_size / subset_size)*log2(right_branch_size / subset_size), (float)0, right_branch_size <= 0)
        );

    return select(info_gain / split_info, (float)0, split_info <= 9e-7);
}

__kernel void gain_ratio(
    __global float *dataset,  int n_objects,  int n_attributes,
    __global int *subset_index,  int attribute_index,
    int n_candidates,  __global float *candidates,
    int n_classes) {

    const int idx = get_global_id(0);

    // only one thread must compute the subset entropy, since
    // its value is shared across every other calculation
    __local float subset_entropy;
    if(idx % LOCAL_WORK_GROUP_SIZE == 0) {
        subset_entropy = entropy_by_index(dataset, n_objects, n_attributes, subset_index, n_classes);
    }
    barrier(CLK_GLOBAL_MEM_FENCE);  // globally block threads

    if(idx < n_candidates) {
        candidates[idx] = device_gain_ratio(
            dataset, n_objects, n_attributes,
            subset_index, attribute_index,
            candidates[idx],
            n_classes,
            subset_entropy
        );
    }
}


__kernel void predict(
    __global float *dataset, int n_objects, int n_attributes,
    __global float *tree, int n_data,
    int n_predictions, __global int *predictions) {

    const int idx = get_global_id(0);

    if (idx < n_predictions) {
        float current_node = 0;
        while(TRUE) {
            float terminal = at(tree, n_data, current_node, TERM);

            if(terminal) {
                predictions[idx] = (int)at(tree, n_data, current_node, ATTR);
                break;
            }

            float attribute, threshold;

            attribute = at(tree, n_data, current_node, ATTR);
            threshold = at(tree, n_data, current_node, THRES);

            if(at(dataset, n_attributes, idx, attribute) > threshold) {
                current_node = at(tree, n_data, current_node, RIGHT);
            } else {
                current_node = at(tree, n_data, current_node, LEFT);
            }
        }
    }
}