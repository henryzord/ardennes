//__device__ float get_value_at(float *dataset, int n_attributes, int object_index, int attribute_index) {
//    return dataset[object_index * n_attributes + attribute_index];
//}

__device__ float entropy_by_index(float *dataset, int n_objects, int n_attributes, int *subset_index, int n_classes, float *class_labels) {
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
        l_entropy += ((count_class > 0) && (subset_size > 0)) * ((count_class / subset_size) * log2f(count_class / subset_size));
    }

    return -l_entropy;
}

__device__ float information_gain(
    float *dataset, int n_objects, int n_attributes,
    int *subset_index, int attribute_index,
    float candidate,
    int n_classes, float *class_labels) {

    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    __shared__ float subset_entropy;
    // only one thread must compute the subset entropy, since its value is shared across every other calculation
    if(idx == 0) {
        subset_entropy = entropy_by_index(dataset, n_objects, n_attributes, subset_index, n_classes, class_labels);
    }
   __syncthreads();

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
        subset_size = 0, is_from_class =0, is_left = 0;

        for(j = 0; j < n_objects; j++) {
            is_left = (float)(dataset[j * n_attributes + attribute_index] < candidate);
            is_from_class = (float)(abs(dataset[j * n_attributes + class_index] - class_labels[k]) < 0.01);

            left_branch_size += is_left;
            right_branch_size += !is_left;

            left_count_class += is_left * is_from_class;
            right_count_class += !is_left * is_from_class;

            subset_size += (float)(subset_index[j]);
        }
        left_entropy += (left_branch_size > 0 && left_count_class > 0)*((left_count_class / left_branch_size) * log2f(left_count_class / left_branch_size));
        right_entropy += (right_branch_size > 0 && right_count_class > 0)*((right_count_class / right_branch_size) * log2f(right_count_class / right_branch_size));
    }

    sum_term =
        ((left_branch_size / subset_size) * -left_entropy) +
        ((right_branch_size / subset_size) * -right_entropy);

    return subset_entropy - sum_term;
}

__device__ float device_gain_ratio(
    float *dataset, int n_objects, int n_attributes,
    int *subset_index, int attribute_index,
    float candidate,
    int n_classes, float *class_labels) {

    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    __shared__ float subset_entropy;
    // only one thread must compute the subset entropy, since
    // its value is shared across every other calculation
    if(idx == 0) {
        subset_entropy = entropy_by_index(dataset, n_objects, n_attributes, subset_index, n_classes, class_labels);
    }
   __syncthreads();

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
            is_left = (float)dataset[j * n_attributes + attribute_index] < candidate;
            is_from_class = (float)(abs(dataset[j * n_attributes + class_index] - class_labels[k]) < 0.01);

            left_branch_size += is_left * subset_index[j];
            right_branch_size += !is_left * subset_index[j];

            left_count_class += is_left * subset_index[j] * is_from_class;
            right_count_class += !is_left * subset_index[j] * is_from_class;

            subset_size += (float)(subset_index[j]);
        }
        left_entropy += ((left_branch_size > 0) && (left_count_class > 0))*((left_count_class / left_branch_size) * log2f(left_count_class / left_branch_size));
        right_entropy += ((right_branch_size > 0) && (right_count_class > 0))*((right_count_class / right_branch_size) * log2f(right_count_class / right_branch_size));
    }

    sum_term =
        ((left_branch_size / subset_size) * -left_entropy) +
        ((right_branch_size / subset_size) * -right_entropy);

    float
        info_gain = subset_entropy - sum_term,
        split_info = -(
            (left_branch_size > 0)*((left_branch_size / subset_size)*log2f(left_branch_size / subset_size)) +
            (right_branch_size > 0)*((right_branch_size / subset_size)*log2f(right_branch_size / subset_size))
        );

    return (split_info > 0) * (info_gain / split_info);
}


__global__ void gain_ratio(
    float *dataset, int n_objects, int n_attributes,
    int *subset_index, int attribute_index,
    int n_candidates, float *candidates,
    int n_classes, float *class_labels) {

    const int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if(idx < n_candidates) {
        candidates[idx] = device_gain_ratio(
            dataset, n_objects, n_attributes,
            subset_index, attribute_index,
            candidates[idx],
            n_classes, class_labels
        );
    }
}