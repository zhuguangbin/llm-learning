import json

def filter_invalid_conversations(input_filename, output_filename):
    with open(input_filename, 'r') as infile, open(output_filename, 'w') as outfile:
        data = json.load(infile)
	# 过滤掉conversations为空的数据
        valid_data = [d for d in data if d['conversations']]

        new_data = []
        
        for record in valid_data:
            valid_record = True
            new_conversations = []

	    # 过滤掉conversation中from值不合法的记录
            for conversation in record["conversations"]:
                if conversation["from"] not in ["human", "gpt"]:
                    valid_record = False
                    break
                new_conversations.append(conversation)
            
            if valid_record:
                record["conversations"] = new_conversations
                new_data.append(record)

        json.dump(new_data, outfile, ensure_ascii=False, indent=4)

# 使用函数
filter_invalid_conversations('ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json', 'ShareGPT_filtered.json')
