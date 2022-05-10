def smart_truncate(content, length=512, suffix=""):
    if len(content) <= length:
        return content
    return " ".join(content[: length + 1].split(" ")[0:-1]) + suffix
