use serde_json::Value;

const INPUT_SIZE: usize = 768;

fn weight(weight_relation: &[Value], weight_array: &mut [i16], stride: usize, k: i32, flip: bool) {
    #![allow(clippy::cast_possible_truncation)]
    for (i, output) in weight_relation.iter().enumerate() {
        for (j, weight) in output.as_array().unwrap().iter().enumerate() {
            let index = if flip { j * stride + i } else { i * stride + j };
            let value = weight.as_f64().unwrap();
            weight_array[index] = (value * f64::from(k)) as i16;
        }
    }
}

fn bias(bias_relation: &[Value], bias_array: &mut [i16], k: i32) {
    #![allow(clippy::cast_possible_truncation)]
    for (i, bias) in bias_relation.iter().enumerate() {
        let value = bias.as_f64().unwrap();
        bias_array[i] = (value * f64::from(k)) as i16;
    }
}

type FourVecs = (Vec<i16>, Vec<i16>, Vec<i16>, Vec<i16>);

pub fn from_json(
    json: &str,
    qa: i32,
    qb: i32,
    ft_name: &str,
    out_name: &str,
) -> Result<FourVecs, Box<dyn std::error::Error>> {
    let json: Value = serde_json::from_str(json)?;
    let object = json.as_object().ok_or("Input JSON is not an object")?;

    // make keys for the json object
    let ft_weight_key = format!("{ft_name}.weight");
    let ft_bias_key = format!("{ft_name}.bias");
    let out_weight_key = format!("{out_name}.weight");
    let out_bias_key = format!("{out_name}.bias");

    if object.len() > 4 {
        return Err(format!(
            "Too many fields in JSON, expected 4 but got {}",
            object.len()
        )
        .into());
    }
    if object.len() < 4 {
        return Err(format!(
            "Too few fields in JSON, expected 4 but got {}",
            object.len()
        )
        .into());
    }

    // extract fields from the json object
    let ft_weights = object.get(&ft_weight_key).ok_or_else(|| {
        format!(
            "{} not found, keys of object are: {:?}.\nmaybe try passing --ft-name {}?",
            ft_weight_key,
            object.keys().collect::<Vec<_>>(),
            object.keys().next().unwrap().split_once('.').unwrap().0
        )
    })?;
    let ft_bias = object.get(&ft_bias_key).ok_or_else(|| {
        format!(
            "{} not found, keys of object are: {:?}.\nmaybe try passing --ft-name {}?",
            ft_bias_key,
            object.keys().collect::<Vec<_>>(),
            object.keys().next().unwrap().split_once('.').unwrap().0
        )
    })?;
    let out_weights = object.get(&out_weight_key).ok_or_else(|| {
        format!(
            "{} not found, keys of object are: {:?}.\nmaybe try passing --out-name {}?",
            out_weight_key,
            object.keys().collect::<Vec<_>>(),
            object.keys().nth(3).unwrap().split_once('.').unwrap().0
        )
    })?;
    let out_bias = object.get(&out_bias_key).ok_or_else(|| {
        format!(
            "{} not found, keys of object are: {:?}.\nmaybe try passing --out-name {}?",
            out_bias_key,
            object.keys().collect::<Vec<_>>(),
            object.keys().nth(3).unwrap().split_once('.').unwrap().0
        )
    })?;

    // check that the fields are arrays
    let ft_weights = ft_weights
        .as_array()
        .ok_or("perspective.weight is not an array")?;
    let ft_bias = ft_bias
        .as_array()
        .ok_or("perspective.bias is not an array")?;
    let out_weights = out_weights.as_array().ok_or("out.weight is not an array")?;
    let out_bias = out_bias.as_array().ok_or("out.bias is not an array")?;

    // check that the arrays are the right size
    let ft_neurons = ft_weights.len();
    println!("ft_size: {ft_neurons}");
    let out_size = out_weights[0]
        .as_array()
        .ok_or("out.weight[0] is not an array")?
        .len();
    println!("out_size: {out_size}");
    if 2 * ft_neurons != out_size {
        return Err(format!(
            "perspective.weight has {ft_neurons} neurons, but out.weight has {out_size} inputs (should be twice as many)"
        ).into());
    }

    println!("Hope you're using a {ft_neurons}x2 net, because that's what this looks like to me!");

    // allocate buffers for the weights and biases
    let mut feature_weights_buf = vec![0i16; ft_neurons * INPUT_SIZE];
    let mut feature_bias_buf = vec![0i16; ft_neurons];
    let mut output_weights_buf = vec![0i16; out_size];
    let mut output_bias_buf = vec![0i16; 1];

    // read the weights and biases into the buffers
    weight(ft_weights, &mut feature_weights_buf, ft_neurons, qa, true);
    bias(ft_bias, &mut feature_bias_buf, qa);
    weight(out_weights, &mut output_weights_buf, out_size, qb, false);
    bias(out_bias, &mut output_bias_buf, qa * qb);

    Ok((
        feature_weights_buf,
        feature_bias_buf,
        output_weights_buf,
        output_bias_buf,
    ))
}

mod tests {
    #[test]
    fn test_from_json_0030() {
        let json = std::fs::read_to_string("validation/net0030/viri0030.json").unwrap();
        let (ft_weights, ft_bias, out_weights, out_bias) =
            crate::convert::from_json(&json, 255, 64, "ft", "out").unwrap();
        assert_eq!(ft_weights.len(), 768 * 256);
        assert_eq!(ft_bias.len(), 256);
        assert_eq!(out_weights.len(), 512);
        assert_eq!(out_bias.len(), 1);
        let validation_ft_weight = std::fs::read("validation/net0030/feature_weights.bin").unwrap();
        let validation_ft_bias = std::fs::read("validation/net0030/feature_bias.bin").unwrap();
        let validation_o_weight = std::fs::read("validation/net0030/output_weights.bin").unwrap();
        let validation_o_bias = std::fs::read("validation/net0030/output_bias.bin").unwrap();
        let validation_binary = std::fs::read("validation/net0030/viri0030.bin").unwrap();

        let validation_concat = [
            &validation_ft_weight[..],
            &validation_ft_bias[..],
            &validation_o_weight[..],
            &validation_o_bias[..],
        ]
        .concat();

        assert_eq!(
            validation_binary, validation_concat,
            "validation binary is not as expected"
        );

        let our_concat = [
            &ft_weights[..],
            &ft_bias[..],
            &out_weights[..],
            &out_bias[..],
        ]
        .concat();
        let our_concat = unsafe {
            let inner = our_concat.align_to::<u8>().1;
            inner
        };

        assert_eq!(
            validation_binary, our_concat,
            "our binary is not as expected"
        );
    }

    #[test]
    fn test_from_json_0056() {
        let json = std::fs::read_to_string("validation/net0056/viri0056.json").unwrap();
        let (ft_weights, ft_bias, out_weights, out_bias) =
            crate::convert::from_json(&json, 255, 64, "perspective", "out").unwrap();
        assert_eq!(ft_weights.len(), 768 * 512);
        assert_eq!(ft_bias.len(), 512);
        assert_eq!(out_weights.len(), 1024);
        assert_eq!(out_bias.len(), 1);
        let validation_ft_weight = std::fs::read("validation/net0056/feature_weights.bin").unwrap();
        let validation_ft_bias = std::fs::read("validation/net0056/feature_bias.bin").unwrap();
        let validation_o_weight = std::fs::read("validation/net0056/output_weights.bin").unwrap();
        let validation_o_bias = std::fs::read("validation/net0056/output_bias.bin").unwrap();
        let validation_binary = std::fs::read("validation/net0056/viri0056.bin").unwrap();

        let validation_concat = [
            &validation_ft_weight[..],
            &validation_ft_bias[..],
            &validation_o_weight[..],
            &validation_o_bias[..],
        ]
        .concat();

        assert_eq!(
            validation_binary, validation_concat,
            "validation binary is not as expected"
        );

        let our_concat = [
            &ft_weights[..],
            &ft_bias[..],
            &out_weights[..],
            &out_bias[..],
        ]
        .concat();
        let our_concat = unsafe {
            let inner = our_concat.align_to::<u8>().1;
            inner
        };

        assert_eq!(
            validation_binary, our_concat,
            "our binary is not as expected"
        );
    }
}
