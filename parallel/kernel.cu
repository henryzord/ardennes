//__device__ float get_value_at(float *dataset, int n_attributes, int object_index, int attribute_index) {
//    return dataset[object_index * n_attributes + attribute_index];
//}

__device__ float entropy(float *dataset, int n_objects, int n_attributes, int *subset_index, float *class_labels, int n_classes) {
    const int class_index = n_attributes - 1;

    int i, j;

    float entropy = 0;

    float subset_size = 0;
    for(j = 0; j < n_objects; j++) {
        subset_size += (float)subset_index[i];
    }

    for(i = 0; i < n_classes; i++) {
        float count_class = 0;
        for(j = 0; j < n_objects; j++) {
            count_class += (float)(class_labels[i] == dataset[j * n_attributes + class_index]);

        }
        entropy += (count_class / subset_size) * log2f(count_class / subset_size);
    }

    return -entropy;
}

__device__ float information_gain(
    float *dataset, int n_objects, int n_attributes,
    int *subset_index, int attribute_index, float candidate, float *class_labels, int n_classes) {

    const int class_index = n_attributes - 1;

    int j, k;

    float   left_entropy = 0, right_entropy = 0, left_branch_size = 0, right_branch_size = 0,
            subset_size = 0, left_count_class, right_count_class, is_left, is_from_class;
    for(k=0; k < n_classes; k++) {
        left_count_class = 0; right_count_class = 0; left_branch_size = 0; right_branch_size = 0; subset_size = 0;
        for(j=0; j < n_objects; j++) {
            is_left = (float)(dataset[j * n_attributes + attribute_index] < candidate);
            is_from_class = (float)(dataset[j * n_attributes + class_index] == class_labels[k]);

            left_branch_size += is_left;
            right_branch_size += !is_left;

            left_count_class += is_left * is_from_class;
            right_count_class += !(is_left) * is_from_class;

            subset_size += (float)(subset_index[j]);
        }
        left_entropy += (left_count_class / left_branch_size) * log2f(left_count_class / left_branch_size);
        right_entropy += (right_count_class / right_branch_size) * log2f(right_count_class / right_branch_size);
    }

    float sum_term =
        ((left_branch_size / subset_size) * -left_entropy) +
        ((right_branch_size / subset_size) * -right_entropy);

//    return entropy(dataset, n_objects, n_attributes, subset_index, class_labels, n_classes)  - sum_term;
    return candidate;
}

//def host_information_gain(self, subset, subset_left, subset_right, target_attr):
//    sum_term = 0.
//    for child_subset in [subset_left, subset_right]:
//        sum_term += (child_subset.shape[0] / float(subset.shape[0])) * self.host_entropy(child_subset, target_attr)
//
//    ig = self.host_entropy(subset, target_attr) - sum_term
//    return ig


__global__ void gain_ratio(
    float *dataset, int n_objects, int n_attributes,
    int *subset_index, int attribute_index, int n_candidates, float *candidates,
    float *class_labels, int n_classes, float *out) {

    const int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if(idx < n_candidates) {
        out[idx] = -1;
//        candidates[idx] = information_gain(dataset, n_objects, n_attributes,
//        subset_index, attribute_index, candidates[idx], class_labels, n_classes);
    }
}