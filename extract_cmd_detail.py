import json
import sys
sys.stdout.reconfigure(encoding='utf-8')
with open('fprime_dictionary.json', 'r', encoding='utf-8') as f:
    d = json.load(f)
    cmds = d.get('commands', [])
    for c in cmds:
        name = c.get('name', 'Unknown')
        opcode = c.get('opcode', 'Unknown')
        annotation = c.get('annotation', '')
        params = c.get('formalParams', [])
        print(f"### {name}")
        print(f"- **Opcode:** {opcode}")
        if annotation:
            print(f"- **Mô tả:** {annotation}")
        if params:
            print("- **Tham số (Arguments):**")
            for p in params:
                pname = p.get('name', '')
                ptype = p.get('type', {}).get('name', '')
                pkind = p.get('type', {}).get('kind', '')
                print(f"  - `{pname}` (Kiểu: {ptype} - {pkind})")
        else:
            print("- **Tham số:** (Không có)")
        print()
