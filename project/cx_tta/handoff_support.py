"""Representation-independent contact reads for privileged diagnostics."""
def matching_links(contacts,object_name,links):
    pairs={tuple(sorted(pair)) for pair in contacts}
    return [link for link in links if tuple(sorted((object_name,link))) in pairs]
