int next_node(int current_node, int go_left, int finished) {
    return (!finished * ((current_node * 2) + 1 + go_left)) + (finished * current_node);
}

__kernel void predict_objects(
    __global float *dataset, int n_objects, int n_attributes,
    __global int *attribute_index, __global float *thresholds,
    int n_individuals, int H, __global int *predictions) {

    const int idx = get_global_id(0);
    if (idx < n_individuals) {
        int i, h, current_node, go_left, finished;

        for(i = 0; i < n_objects; i++) {
            current_node = 0, finished = 0;
            
            for(h = 0; h < H; h++) {
                finished = finished || (attribute_index[current_node] == (n_attributes - 1));
                go_left = dataset[i * n_objects + attribute_index[current_node]] <= thresholds[current_node]; 
                current_node = next_node(current_node, go_left, finished);
            }
            predictions[idx * n_individuals + i] = (int)thresholds[current_node];
        }
    }
}


float entropy_by_index(__global float *dataset, int n_objects, int n_attributes, __global int *subset_index, int n_classes, __global float *class_labels) {
    const int class_index = n_attributes - 1;

    int i, j;

    float l_entropy = 0, subset_size = 0, count_class = 0;

    for(j = 0; j < n_objects; j++) {
        subset_size += (float)subset_index[j];
    }

    for(i = 0; i < n_classes; i++) {
        count_class = 0;
        for(j = 0; j < n_objects; j++) {
            count_class += (float)(subset_index[j] * (class_labels[i] == dataset[j * n_attributes + class_index]));

        }
        l_entropy += select((count_class / subset_size) * log2(count_class / subset_size), (float)0, count_class <= 0);
    }

    return -l_entropy;
}

float device_gain_ratio(
    __global float *dataset, int n_objects, int n_attributes,
    __global int *subset_index, int attribute_index,
    float candidate,
    int n_classes, __global float *class_labels, float subset_entropy) {

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
            is_from_class = (float)(fabs(dataset[j * n_attributes + class_index] - class_labels[k]) < 0.01);

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

    return select(info_gain / split_info, (float)0, info_gain <= 0);

}

__kernel void gain_ratio(__global float *dataset,  int n_objects,  int n_attributes,
     __global int *subset_index,  int attribute_index,
     int n_candidates,  __global float *candidates,
     int n_classes,  __global float *class_labels) {

    const int idx = get_global_id(0);

    // only one thread must compute the subset entropy, since
    // its value is shared across every other calculation
    __local float subset_entropy;
    if(idx == 0) {
        subset_entropy = entropy_by_index(dataset, n_objects, n_attributes, subset_index, n_classes, class_labels);
    }
    barrier(CLK_GLOBAL_MEM_FENCE);  // globally block threads

    if(idx < n_candidates) {
        candidates[idx] = device_gain_ratio(
            dataset, n_objects, n_attributes,
            subset_index, attribute_index,
            candidates[idx],
            n_classes, class_labels,
            subset_entropy
        );
    }

}